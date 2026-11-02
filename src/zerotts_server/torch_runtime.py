"""torch-cuda runtime: ZeroTTS frame generation in PyTorch on one GPU (ADR 0007).

A port of the ggml runtime in references/ZeroTTS-repo/cpp/src/zerotts.cpp, which reproduces the
ONNX frame codes, reading models/ZeroTTS-torch/zerotts.safetensors (the GGUF f32 tensors under
their GGUF names). This slice runs eager fp32; the sampler mirrors local_frame_decode.onnx op for
op so that equal random draws give equal codes.

Layout: weights are (out, in) as nn.Linear stores them. The global decoder's self-attention KV
cache is preallocated for the voice block, <soa> and MAX_FRAMES frames and written by position:
the voice block at 0..V-1, <soa> at V, frame t at V + 1 + t. The text's cross-attention K/V are
projected once per segment. The local (depth) transformer has no RoPE and a 17-slot cache.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from zerotts_server.model.tokenizer import load_tokenizer

from .config import Sampling
from .runtime import EOA_EXTRA_FRAMES, MAX_FRAMES, MIN_FRAMES, draw_u

# local_frame_decode.onnx's clips.
MIN_TEMPERATURE = 1e-4
MIN_TOPP = 1e-6
MAX_U = 0.999999


def resolve_device(device: str) -> torch.device:
    """The CUDA device to run on; raises instead of falling back to CPU."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("runtime torch-cuda needs a cuda device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"{device} does not exist")
    return torch.device("cuda", index)


class TorchZeroTTS:
    """One worker's ZeroTTS: weights, caches and the frame loop on one GPU, batch 1."""

    def __init__(self, model_dir: str | Path, device: str):
        model_dir = Path(model_dir)
        self.device = resolve_device(device)
        config = json.loads((model_dir / "config.json").read_text())
        special = config["special_tokens"]
        self.soa_id, self.slot_id, self.eoa_id = (
            int(special[name]) for name in ("<soa>", "<slot>", "<eoa>")
        )
        self.tokenizer = load_tokenizer(config, model_dir)

        path = model_dir / "zerotts.safetensors"
        with safe_open(path, "pt") as f:
            hp = json.loads(f.metadata()["gguf"])
        self.d_model = hp["zerotts.d_model"]
        self.n_heads = hp["zerotts.n_heads"]
        self.n_layers = hp["zerotts.n_layers"]
        self.local_n_layers = hp["zerotts.local_n_layers"]
        self.local_n_heads = hp["zerotts.local_n_heads"]
        self.n_depth = hp["zerotts.n_depth"]
        self.num_codebooks = hp["zerotts.num_codebooks"]
        self.codebook_size = hp["zerotts.codebook_size"]
        self.n_voice_queries = hp["zerotts.n_voice_queries"]
        self.cross_qk_norm = bool(hp["zerotts.cross_qk_norm"])
        self.ln_eps = hp["zerotts.layer_norm_eps"]
        self.qk_eps = hp["zerotts.qk_norm_eps"]
        if self.n_depth != self.num_codebooks + 1:
            raise ValueError("local.n_depth must be num_codebooks + 1")

        w = load_file(path, device=str(self.device))
        self.w = w
        self.text_embed = w["text_embed.weight"]
        self.depth_embed = w["local.depth_embed"][0]  # (n_depth, d)
        self.audio_embed = torch.stack(
            [w[f"audio_embed.{c}.weight"] for c in range(self.num_codebooks)]
        )  # (K, C + 1, d)
        self.audio_head = [w[f"audio_head.{c}.weight"] for c in range(self.num_codebooks)]
        ids = torch.tensor([self.slot_id, self.eoa_id, self.soa_id], device=self.device)
        self.ctrl_rows = self.text_embed[ids[:2]]  # control head, tied to the text embedding
        self.slot_embed = self.text_embed[ids[0]]
        self.soa_embed = self.text_embed[ids[2]]
        self.codebooks = torch.arange(self.num_codebooks, device=self.device)

        # RoPE (adjacent pairs, ggml's GGML_ROPE_TYPE_NORMAL) for every position either attention
        # side can reach: text tokens (512 + BOS/EOT) and decoder positions.
        d_head = self.d_model // self.n_heads
        self.n_ctx = self.n_voice_queries + 1 + MAX_FRAMES
        positions = torch.arange(max(self.n_ctx, 514), dtype=torch.float64)
        inv_freq = hp["zerotts.rope_theta"] ** (
            -torch.arange(0, d_head, 2, dtype=torch.float64) / d_head
        )
        angles = positions[:, None] * inv_freq[None, :]
        self.rope_cos = angles.cos().float().to(self.device)
        self.rope_sin = angles.sin().float().to(self.device)

        self.k_cache = torch.zeros(
            self.n_layers, self.n_heads, self.n_ctx, d_head, device=self.device
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        local_d_head = self.d_model // self.local_n_heads
        self.local_k = torch.zeros(
            self.local_n_layers, self.local_n_heads, self.n_depth, local_d_head, device=self.device
        )
        self.local_v = torch.zeros_like(self.local_k)
        self.cross: list[tuple[torch.Tensor, torch.Tensor]] = []

    # ── building blocks ──────────────────────────────────────────────────────

    def _ln(self, x, name):
        return F.layer_norm(
            x, (self.d_model,), self.w[f"{name}.weight"], self.w[f"{name}.bias"], self.ln_eps
        )

    def _ffn(self, x, prefix):
        w = self.w
        x = F.silu(F.linear(x, w[f"{prefix}.ffn_up.weight"], w[f"{prefix}.ffn_up.bias"]))
        return F.linear(x, w[f"{prefix}.ffn_down.weight"], w[f"{prefix}.ffn_down.bias"])

    def _heads(self, x, n_heads):
        """(T, d) -> (H, T, d_head)."""
        return x.view(x.shape[0], n_heads, -1).transpose(0, 1)

    def _rope(self, x, start: int):
        """Rotate (H, T, d_head) at positions start..start+T-1."""
        cos = self.rope_cos[start : start + x.shape[1]]
        sin = self.rope_sin[start : start + x.shape[1]]
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)

    def _rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.qk_eps)

    def _attend(self, q, k, v, out_weight):
        """q (H, Tq, dh) over k, v (H, Tk, dh), no mask; -> (Tq, d) through the output proj."""
        o = F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
        return F.linear(o.transpose(0, 1).reshape(q.shape[1], self.d_model), out_weight)

    def _self_attn(self, x, layer: int, start: int):
        """Global decoder self-attention: write positions start.. to the cache, attend 0..end."""
        p = f"dec.{layer}.self"
        w = self.w
        end = start + x.shape[0]
        q = self._rope(self._heads(F.linear(x, w[f"{p}_q.weight"]), self.n_heads), start)
        k = self._rope(self._heads(F.linear(x, w[f"{p}_k.weight"]), self.n_heads), start)
        self.k_cache[layer, :, start:end] = k
        self.v_cache[layer, :, start:end] = self._heads(
            F.linear(x, w[f"{p}_v.weight"]), self.n_heads
        )
        return self._attend(
            q, self.k_cache[layer, :, :end], self.v_cache[layer, :, :end], w[f"{p}_o.weight"]
        )

    def _decoder_step(self, x, position: int):
        """One global decoder position (<soa> or a frame); returns the final-norm hidden (1, d)."""
        w = self.w
        for i in range(self.n_layers):
            p = f"dec.{i}"
            x = x + self._self_attn(self._ln(x, f"{p}.norm_self"), i, position)
            q = F.linear(self._ln(x, f"{p}.norm_cross"), w[f"{p}.cross_q.weight"])
            q = self._heads(q, self.n_heads)
            if self.cross_qk_norm:
                q = self._rms(q)
            k, v = self.cross[i]
            x = x + self._attend(self._rope(q, position), k, v, w[f"{p}.cross_o.weight"])
            x = x + self._ffn(self._ln(x, f"{p}.norm_ffn"), p)
        return self._ln(x, "ln_f")

    def _local_step(self, x, depth: int):
        """One depth position of the frame decoder; returns the final-norm hidden (1, d)."""
        w = self.w
        x = x + self.depth_embed[depth]
        for i in range(self.local_n_layers):
            p = f"loc.{i}"
            a = self._ln(x, f"{p}.norm_attn")
            q = self._heads(F.linear(a, w[f"{p}.attn_q.weight"]), self.local_n_heads)
            self.local_k[i, :, depth] = F.linear(a, w[f"{p}.attn_k.weight"]).view(
                self.local_n_heads, -1
            )
            self.local_v[i, :, depth] = F.linear(a, w[f"{p}.attn_v.weight"]).view(
                self.local_n_heads, -1
            )
            x = x + self._attend(
                q,
                self.local_k[i, :, : depth + 1],
                self.local_v[i, :, : depth + 1],
                w[f"{p}.attn_o.weight"],
            )
            x = x + self._ffn(self._ln(x, f"{p}.norm_ffn"), p)
        return self._ln(x, "local.ln_f")

    @staticmethod
    def _sample(scaled, u, topk: int, topp: float):
        """local_frame_decode.onnx's sampler on temperature-scaled logits (n,); returns an index.

        Sort, keep the top k, drop ranks whose preceding mass exceeds topp, renormalize, then
        take the first rank whose CDF passes u.
        """
        values, order = torch.sort(scaled, descending=True, stable=True)
        n = values.shape[0]
        values = values.masked_fill(torch.arange(n, device=values.device) >= topk, -torch.inf)
        p = torch.softmax(values, dim=-1)
        values = values.masked_fill(torch.cumsum(p, dim=-1) - p > topp, -torch.inf)
        cdf = torch.cumsum(torch.softmax(values, dim=-1), dim=-1)
        rank = (cdf > u.clamp(0.0, MAX_U)).to(torch.int32).argmax()
        return order[rank]

    # ── frame loop (the ONNX runtime's three graphs, as in ZeroTTS._generate_frames) ─────────

    @torch.no_grad()
    def begin(self, text_ids: np.ndarray, voice_emb: np.ndarray) -> torch.Tensor:
        """Text encoder, cross K/V, voice prefix and <soa>; returns the hidden for frame 0."""
        w = self.w
        ids = torch.as_tensor(np.asarray(text_ids, dtype=np.int64), device=self.device)
        x = self.text_embed[ids]
        for i in range(self.n_layers):
            p = f"txt.{i}"
            a = self._ln(x, f"{p}.norm_attn")
            q = self._rope(self._heads(F.linear(a, w[f"{p}.attn_q.weight"]), self.n_heads), 0)
            k = self._rope(self._heads(F.linear(a, w[f"{p}.attn_k.weight"]), self.n_heads), 0)
            v = self._heads(F.linear(a, w[f"{p}.attn_v.weight"]), self.n_heads)
            x = x + self._attend(q, k, v, w[f"{p}.attn_o.weight"])
            x = x + self._ffn(self._ln(x, f"{p}.norm_ffn"), p)
        states = self._ln(x, "text_ln_f")
        self.cross = []
        for i in range(self.n_layers):
            p = f"dec.{i}"
            context = self._ln(states, f"{p}.norm_context")
            k = self._heads(F.linear(context, w[f"{p}.cross_k.weight"]), self.n_heads)
            if self.cross_qk_norm:
                k = self._rms(k)
            v = self._heads(F.linear(context, w[f"{p}.cross_v.weight"]), self.n_heads)
            self.cross.append((self._rope(k, 0), v))

        # The voice block: bidirectional among itself, no cross-attention (zerotts.cpp).
        voice = torch.as_tensor(np.asarray(voice_emb, dtype=np.float32), device=self.device)
        x = voice.reshape(self.n_voice_queries, self.d_model)
        for i in range(self.n_layers):
            p = f"dec.{i}"
            x = x + self._self_attn(self._ln(x, f"{p}.norm_self"), i, 0)
            x = x + self._ffn(self._ln(x, f"{p}.norm_ffn"), p)
        return self._decoder_step(self.soa_embed[None], self.n_voice_queries)

    @torch.no_grad()
    def frame(self, h, forbid_eoa: bool, sampling: Sampling, ctrl_u, audio_u, seen, forced=None):
        """Decode one frame from the global hidden h (1, d).

        Returns device tensors (is_eoa, codes (K,), scaled control logits (2,), scaled audio
        logits (K, C)); the logits are what the sampler sees, after penalty and temperature.
        `forced` (K,), as zerotts_frame_forced, replaces the sampled codes on the depth
        decoder's input path while `codes` still reports the sampled ones (teacher forcing).
        """
        ctrl = F.linear(self._local_step(h, 0), self.ctrl_rows)[0]
        if forbid_eoa:
            ctrl = ctrl.clone()
            ctrl[1] = torch.finfo(ctrl.dtype).min
        ctrl = ctrl / max(sampling.text_temperature, MIN_TEMPERATURE)
        text_topk = sampling.text_topk if sampling.text_topk > 0 else self.codebook_size
        ctrl_u = torch.as_tensor(ctrl_u, device=self.device).reshape(())
        is_eoa = self._sample(ctrl, ctrl_u, min(max(text_topk, 1), 2), 1.0) == 1

        audio_u = torch.as_tensor(audio_u, device=self.device).reshape(-1)
        audio_topk = sampling.audio_topk if sampling.audio_topk > 0 else self.codebook_size
        audio_topk = min(max(audio_topk, 1), self.codebook_size)
        topp = min(max(sampling.audio_topp, MIN_TOPP), 1.0)
        penalty = sampling.audio_repetition_penalty
        temperature = max(sampling.audio_temperature, MIN_TEMPERATURE)
        # Depth 1's input is always <slot>; the audio channels never see the control token.
        x = self.slot_embed[None]
        codes, scaled = [], []
        for c in range(self.num_codebooks):
            logits = F.linear(self._local_step(x, c + 1), self.audio_head[c])[0]
            logits = torch.where(
                seen[c], torch.where(logits < 0, logits * penalty, logits / penalty), logits
            )
            logits = logits / temperature
            code = self._sample(logits, audio_u[c], audio_topk, topp)
            codes.append(code)
            scaled.append(logits)
            fed = code[None] if forced is None else forced[c : c + 1]
            x = self.audio_embed[c].index_select(0, fed)
        return is_eoa, torch.stack(codes), ctrl, torch.stack(scaled)

    @torch.no_grad()
    def advance(self, codes, frame_index: int) -> torch.Tensor:
        """Feed frame codes (K,) at position V + 1 + frame_index; returns the next hidden."""
        x = self.slot_embed + self.audio_embed[self.codebooks, codes].sum(0)
        return self._decoder_step(x[None], self.n_voice_queries + 1 + frame_index)

    def frames(self, text, voice_emb, sampling: Sampling, rng: np.random.Generator):
        """ZeroTTS._generate_frames on this model: yields (1, K) int64 codes per frame."""
        if sampling.cfg_scale != 1:
            raise ValueError("runtime torch-cuda supports only cfg_scale = 1")
        h = self.begin(self.tokenizer(text), voice_emb)
        seen = torch.zeros(
            self.num_codebooks, self.codebook_size, dtype=torch.bool, device=self.device
        )
        t = 0
        tail_left = None  # None until <eoa>, then a countdown of tail frames to keep
        while True:
            ctrl_u, audio_u = draw_u(rng, self.num_codebooks)
            is_eoa, codes, _, _ = self.frame(
                h, t < MIN_FRAMES or tail_left is not None, sampling, ctrl_u, audio_u, seen
            )
            seen[self.codebooks, codes] = True
            host = torch.cat([is_eoa.reshape(1).long(), codes]).cpu().numpy()
            if tail_left is None and host[0]:
                tail_left = EOA_EXTRA_FRAMES
            if (tail_left is not None and tail_left <= 0) or t >= MAX_FRAMES:
                break
            yield host[None, 1:]
            if tail_left is not None:
                tail_left -= 1
                if tail_left <= 0:
                    break
            h = self.advance(codes, t)
            t += 1

    def warmup(self) -> None:
        sampling = Sampling()
        h = self.begin(self.tokenizer("a"), np.zeros((1, self.n_voice_queries, self.d_model)))
        seen = torch.zeros(
            self.num_codebooks, self.codebook_size, dtype=torch.bool, device=self.device
        )
        draws = draw_u(np.random.default_rng(0), self.num_codebooks)
        _, codes, _, _ = self.frame(h, True, sampling, *draws, seen)
        self.advance(codes, 0)
        torch.cuda.synchronize(self.device)
