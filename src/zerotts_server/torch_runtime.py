"""torch-cuda runtime: ZeroTTS frame generation in PyTorch on one GPU (ADR 0007).

A port of the ggml runtime in references/ZeroTTS-repo/cpp/src/zerotts.cpp, which reproduces the
ONNX frame codes, reading models/ZeroTTS-torch/zerotts.safetensors (the GGUF f32 tensors under
their GGUF names). The sampler mirrors local_frame_decode.onnx op for op so that, in fp32, equal
random draws give equal codes. The codec is the vendored PyTorch MOSS-Audio-Tokenizer-Nano
(model/moss) on the same GPU, reading codec.safetensors and codec_config.json.

Every shape is fixed so that a segment's start and each frame replay as CUDA Graphs (#21). The
text is padded to MAX_TEXT_TOKENS under a key mask. The global decoder's self-attention KV cache
is preallocated for the voice block, <soa> and every frame and written by position (the voice
block at 0..V-1, <soa> at V, frame t at V + 1 + t) under a mask up to the current position,
which lives on the device. One frame step is the depth decoder's 17 positions, the sampling and
the global decoder step that feeds the sampled codes back. Per frame only the random draws go in
and <eoa> plus the codes come back, through pinned host buffers. Each instance (one worker) has
its own weights, buffers, CUDA stream and graphs. Weights are fp32 on disk; the server computes
in fp16 (the logits are sampled in fp32), and fp32 stays for the parity tests.

Layout: weights are (out, in) as nn.Linear stores them. The local (depth) transformer has no
RoPE and a 17-slot cache.
"""

import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from zerotts_server.model.tokenizer import load_tokenizer

from .config import Sampling
from .engine import STREAM_MAX_CHUNK_FRAMES
from .runtime import EOA_EXTRA_FRAMES, MAX_FRAMES, MIN_FRAMES, draw_u

# local_frame_decode.onnx's clips.
MIN_TEMPERATURE = 1e-4
MIN_TOPP = 1e-6
MAX_U = 0.999999
# The tokenizer's 512 body tokens plus BOS and EOT.
MAX_TEXT_TOKENS = 514
DTYPES = {"float16": torch.float16, "float32": torch.float32}


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
    """One worker's ZeroTTS: weights, buffers, stream and graphs on one GPU, batch 1.

    `dtype` is the compute precision ("float16" or "float32"); `cuda_graph` False runs the same
    fixed-shape code eagerly.
    """

    def __init__(
        self, model_dir: str | Path, device: str, dtype: str = "float16", cuda_graph: bool = True
    ):
        model_dir = Path(model_dir)
        self.device = resolve_device(device)
        self.dtype = DTYPES[dtype]
        self.cuda_graph = cuda_graph
        self.stream = torch.cuda.Stream(self.device)
        self._graphs: dict[str, torch.cuda.CUDAGraph] = {}
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
        K, C, V, d = self.num_codebooks, self.codebook_size, self.n_voice_queries, self.d_model

        dev = self.device
        d_head, local_d_head = d // self.n_heads, d // self.local_n_heads
        w = load_file(path)
        # Fewer kernels per frame: one projection for q, k, v (and the cross K/V), with the
        # attention scale folded into the queries (exact for the global d_head of 64).
        fused = [(f"txt.{i}.attn", d_head) for i in range(self.n_layers)]
        fused += [(f"dec.{i}.self", d_head) for i in range(self.n_layers)]
        fused += [(f"loc.{i}.attn", local_d_head) for i in range(self.local_n_layers)]
        for prefix, head in fused:
            q, k, v = (w.pop(f"{prefix}_{x}.weight") for x in "qkv")
            w[f"{prefix}_qkv.weight"] = torch.cat([q * head**-0.5, k, v])
        for i in range(self.n_layers):
            k, v = (w.pop(f"dec.{i}.cross_{x}.weight") for x in "kv")
            w[f"dec.{i}.cross_kv.weight"] = torch.cat([k, v])
        self.cross_scale = d_head**-0.5
        w = {k: v.to(dev, self.dtype) for k, v in w.items()}
        self.w = w
        self.text_embed = w["text_embed.weight"]
        self.depth_embed = w["local.depth_embed"][0]  # (n_depth, d)
        self.audio_embed = torch.stack(
            [w[f"audio_embed.{c}.weight"] for c in range(K)]
        )  # (K, C + 1, d)
        self.audio_head = [w[f"audio_head.{c}.weight"] for c in range(K)]
        ids = torch.tensor([self.slot_id, self.eoa_id, self.soa_id], device=dev)
        self.ctrl_rows = self.text_embed[ids[:2]]  # control head, tied to the text embedding
        self.slot_embed = self.text_embed[ids[0]]
        self.soa_embed = self.text_embed[ids[2]]
        self.codebooks = torch.arange(K, device=dev)
        self.ranks = torch.arange(C, device=dev)

        # RoPE (adjacent pairs, ggml's GGML_ROPE_TYPE_NORMAL) for every position either attention
        # side can reach: text tokens and decoder positions. One slot past the last frame: the
        # step that samples it also writes its codes' position, unused.
        self.n_ctx = V + 2 + MAX_FRAMES
        positions = torch.arange(max(self.n_ctx, MAX_TEXT_TOKENS), dtype=torch.float64)
        inv_freq = hp["zerotts.rope_theta"] ** (
            -torch.arange(0, d_head, 2, dtype=torch.float64) / d_head
        )
        angles = positions[:, None] * inv_freq[None, :]
        # (positions, d_head) tables for _rotate: cos per pair, sin as (-sin, sin) per pair.
        self.rope_cos = angles.cos().repeat_interleave(2, -1).to(dev, self.dtype)
        self.rope_sin = torch.stack((-angles.sin(), angles.sin()), -1).flatten(-2)
        self.rope_sin = self.rope_sin.to(dev, self.dtype)
        self.positions = torch.arange(self.n_ctx, device=dev)
        self.text_positions = torch.arange(MAX_TEXT_TOKENS, device=dev)
        self.soa_position = torch.tensor([V], device=dev)

        def zeros(*shape, dtype=self.dtype):
            return torch.zeros(*shape, dtype=dtype, device=dev)

        self.k_cache = zeros(self.n_layers, self.n_heads, self.n_ctx, d_head)
        self.v_cache = zeros(self.n_layers, self.n_heads, self.n_ctx, d_head)
        self.cross_k = zeros(self.n_layers, self.n_heads, MAX_TEXT_TOKENS, d_head)
        self.cross_v = zeros(self.n_layers, self.n_heads, MAX_TEXT_TOKENS, d_head)
        self.text_padding = zeros(MAX_TEXT_TOKENS, dtype=torch.bool)
        self.local_k = zeros(self.local_n_layers, self.local_n_heads, self.n_depth, local_d_head)
        self.local_v = zeros(self.local_n_layers, self.local_n_heads, self.n_depth, local_d_head)
        # Segment state: global hidden, frame index, codes seen (repetition penalty).
        self.h = zeros(1, d)
        self.frame_index = zeros(1, dtype=torch.int64)
        self.seen = zeros(K, C, dtype=torch.bool)

        # Graph inputs and outputs, each with a pinned host twin.
        # Per segment: [text length, token ids padded with 0], voice latents, sampling (see
        # _sampling_values). Per frame: [control u, audio u (K), forbid <eoa>] in,
        # [<eoa>, codes (K)] out.
        self._io = {}
        for name, shape, dtype in (
            ("text_in", (1 + MAX_TEXT_TOKENS,), torch.int64),
            ("voice_in", (V, d), torch.float32),
            ("sampling_in", (6,), torch.float32),
            ("frame_in", (K + 2,), torch.float32),
            ("frame_out", (K + 1,), torch.int64),
        ):
            setattr(self, name, zeros(*shape, dtype=dtype))
            host = torch.zeros(shape, dtype=dtype, pin_memory=True)
            self._io[name] = (host, host.numpy())
        torch.cuda.synchronize(dev)

    # ── building blocks ──────────────────────────────────────────────────────

    def _ln(self, x, name):
        return F.layer_norm(
            x, (self.d_model,), self.w[f"{name}.weight"], self.w[f"{name}.bias"], self.ln_eps
        )

    def _ffn(self, x, prefix):
        w = self.w
        x = F.silu(F.linear(x, w[f"{prefix}.ffn_up.weight"], w[f"{prefix}.ffn_up.bias"]))
        return F.linear(x, w[f"{prefix}.ffn_down.weight"], w[f"{prefix}.ffn_down.bias"])

    @staticmethod
    def _rotate(x, cos, sin):
        """RoPE on adjacent pairs of the last dim with _rotate's tables: x·cos + swap(x)·sin."""
        swapped = x.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
        return torch.addcmul(x * cos, swapped, sin)

    def _qkv(self, x, prefix, n_heads, cos=None, sin=None):
        """Fused projection of x (T, d) -> q (prescaled), k, v as (H, T, d_head); q and k
        rotated with (T, d_head) tables when given."""
        qkv = F.linear(x, self.w[f"{prefix}_qkv.weight"]).view(x.shape[0], 3, n_heads, -1)
        qk, v = qkv[:, :2], qkv[:, 2]
        if cos is not None:
            qk = self._rotate(qk, cos[:, None, None], sin[:, None, None])
        return qk[:, 0].transpose(0, 1), qk[:, 1].transpose(0, 1), v.transpose(0, 1)

    def _rms(self, x):
        return F.rms_norm(x, (x.shape[-1],), eps=self.qk_eps)

    def _attend(self, q, k, v, out_weight, masked=None):
        """Prescaled q (H, Tq, dh) over k, v (H, Tk, dh), except keys where masked (Tk,) is
        True; -> (Tq, d) through the output projection.

        Plain matmuls: with a mask, scaled_dot_product_attention picks a kernel that takes
        ~0.1 ms for one query over the full cache.
        """
        scores = torch.matmul(q, k.transpose(-1, -2))
        if masked is not None:
            scores = scores.masked_fill(masked, -torch.inf)
        o = torch.matmul(torch.softmax(scores, dim=-1), v)
        return F.linear(o.transpose(0, 1).reshape(q.shape[1], self.d_model), out_weight)

    def _decoder_step(self, x, position):
        """One global decoder position (device tensor (1,)): writes its K/V to the cache,
        attends to every position up to it and to the text; returns the final-norm hidden."""
        w = self.w
        cos = self.rope_cos.index_select(0, position)
        sin = self.rope_sin.index_select(0, position)
        future = self.positions > position
        for i in range(self.n_layers):
            p = f"dec.{i}"
            q, k, v = self._qkv(self._ln(x, f"{p}.norm_self"), f"{p}.self", self.n_heads, cos, sin)
            self.k_cache[i].index_copy_(1, position, k)
            self.v_cache[i].index_copy_(1, position, v)
            x = x + self._attend(
                q, self.k_cache[i], self.v_cache[i], w[f"{p}.self_o.weight"], future
            )
            q = F.linear(self._ln(x, f"{p}.norm_cross"), w[f"{p}.cross_q.weight"])
            q = q.view(1, self.n_heads, -1)
            if self.cross_qk_norm:
                q = self._rms(q)
            q = self._rotate(q, cos[:, None], sin[:, None]).transpose(0, 1)
            x = x + self._attend(
                q, self.cross_k[i], self.cross_v[i], w[f"{p}.cross_o.weight"], self.text_padding
            )
            x = x + self._ffn(self._ln(x, f"{p}.norm_ffn"), p)
        return self._ln(x, "ln_f")

    def _local_step(self, x, depth: int):
        """One depth position of the frame decoder; returns the final-norm hidden (1, d)."""
        w = self.w
        x = x + self.depth_embed[depth]
        for i in range(self.local_n_layers):
            p = f"loc.{i}"
            q, k, v = self._qkv(self._ln(x, f"{p}.norm_attn"), f"{p}.attn", self.local_n_heads)
            self.local_k[i, :, depth] = k[:, 0]
            self.local_v[i, :, depth] = v[:, 0]
            x = x + self._attend(
                q,
                self.local_k[i, :, : depth + 1],
                self.local_v[i, :, : depth + 1],
                w[f"{p}.attn_o.weight"],
            )
            x = x + self._ffn(self._ln(x, f"{p}.norm_ffn"), p)
        return self._ln(x, "local.ln_f")

    def _sample(self, scaled, u, cut, topp):
        """local_frame_decode.onnx's sampler on temperature-scaled fp32 logits (n,).

        Sort, drop ranks past the top k (`cut`), drop ranks whose preceding mass exceeds topp,
        renormalize, then take the first rank whose CDF passes u (clipped). Returns the index as
        a device tensor.
        """
        values, order = torch.sort(scaled, descending=True, stable=True)
        values = values.masked_fill(cut, -torch.inf)
        p = torch.softmax(values, dim=-1)
        values = values.masked_fill(torch.cumsum(p, dim=-1) - p > topp, -torch.inf)
        cdf = torch.cumsum(torch.softmax(values, dim=-1), dim=-1)
        rank = (cdf > u).to(torch.int32).argmax()
        # index_select, not order[rank]: a 0-d tensor index syncs with the host.
        return order.index_select(0, rank.view(1))[0]

    def _sampling_values(self, sampling: Sampling) -> list[float]:
        """sampling_in: control temperature and top-k, audio temperature, top-k and top-p,
        repetition penalty, clipped as local_frame_decode.onnx's feeds and clips do."""
        C = self.codebook_size
        # The ONNX feeds coerce positive top-k values to int64 before clipping.
        text_topk = sampling.text_topk if sampling.text_topk > 0 else C
        audio_topk = sampling.audio_topk if sampling.audio_topk > 0 else C
        return [
            max(sampling.text_temperature, MIN_TEMPERATURE),
            min(max(int(text_topk), 1), 2),
            max(sampling.audio_temperature, MIN_TEMPERATURE),
            min(max(int(audio_topk), 1), C),
            min(max(sampling.audio_topp, MIN_TOPP), 1.0),
            sampling.audio_repetition_penalty,
        ]

    def _frame(self, h, forbid_eoa, params, ctrl_u, audio_u, seen, forced=None):
        """Decode one frame from the global hidden h (1, d); all arguments are device tensors.

        Returns (is_eoa, codes (K,), scaled control logits (2,), scaled audio logits (K, C)),
        the logits being what the sampler sees, in fp32 after penalty and temperature.
        `forced` (K,), as zerotts_frame_forced, replaces the sampled codes on the depth
        decoder's input path while `codes` still reports the sampled ones (teacher forcing).
        """
        ctrl_u, audio_u = ctrl_u.clamp(0.0, MAX_U), audio_u.clamp(0.0, MAX_U)
        audio_cut = self.ranks >= params[3]
        ctrl = F.linear(self._local_step(h, 0), self.ctrl_rows)[0].float()
        forbidden = forbid_eoa & (self.ranks[:2] == 1)
        ctrl = torch.where(forbidden, torch.finfo(torch.float32).min, ctrl) / params[0]
        is_eoa = self._sample(ctrl, ctrl_u, self.ranks[:2] >= params[1], 1.0) == 1
        penalty = params[5]
        # Depth 1's input is always <slot>; the audio channels never see the control token.
        x = self.slot_embed[None]
        codes, scaled = [], []
        for c in range(self.num_codebooks):
            logits = F.linear(self._local_step(x, c + 1), self.audio_head[c])[0].float()
            logits = torch.where(
                seen[c], torch.where(logits < 0, logits * penalty, logits / penalty), logits
            )
            logits = logits / params[2]
            code = self._sample(logits, audio_u[c], audio_cut, params[4])
            codes.append(code)
            scaled.append(logits)
            fed = code[None] if forced is None else forced[c : c + 1]
            x = self.audio_embed[c].index_select(0, fed)
        return is_eoa, torch.stack(codes), ctrl, torch.stack(scaled)

    def _advance(self, codes, frame_index):
        """Feed frame codes (K,) at position V + 1 + frame_index; returns the next hidden."""
        x = self.slot_embed + self.audio_embed[self.codebooks, codes].sum(0)
        return self._decoder_step(x[None], frame_index + self.n_voice_queries + 1)

    # ── the two graphs: segment start and one frame ─────────────────────────

    def _begin(self) -> None:
        """Text encoder, cross K/V, voice block and <soa> from text_in and voice_in; leaves the
        hidden for frame 0 in h and resets the frame index and seen codes."""
        w = self.w
        n, ids = self.text_in[:1], self.text_in[1:]
        self.text_padding.copy_(self.text_positions >= n)
        cos, sin = self.rope_cos[:MAX_TEXT_TOKENS], self.rope_sin[:MAX_TEXT_TOKENS]
        x = self.text_embed[ids]
        for i in range(self.n_layers):
            p = f"txt.{i}"
            q, k, v = self._qkv(self._ln(x, f"{p}.norm_attn"), f"{p}.attn", self.n_heads, cos, sin)
            x = x + self._attend(q, k, v, w[f"{p}.attn_o.weight"], self.text_padding)
            x = x + self._ffn(self._ln(x, f"{p}.norm_ffn"), p)
        states = self._ln(x, "text_ln_f")
        for i in range(self.n_layers):
            p = f"dec.{i}"
            context = self._ln(states, f"{p}.norm_context")
            kv = F.linear(context, w[f"{p}.cross_kv.weight"]).view(
                MAX_TEXT_TOKENS, 2, self.n_heads, -1
            )
            k = kv[:, 0]
            if self.cross_qk_norm:
                k = self._rms(k)
            # The cross queries are normed, so the attention scale goes on the keys.
            k = self._rotate(k, cos[:, None], sin[:, None]) * self.cross_scale
            self.cross_k[i].copy_(k.transpose(0, 1))
            self.cross_v[i].copy_(kv[:, 1].transpose(0, 1))

        # The voice block at 0..V-1: bidirectional among itself, no cross-attention (zerotts.cpp).
        V = self.n_voice_queries
        cos, sin = self.rope_cos[:V], self.rope_sin[:V]
        x = self.voice_in.to(self.dtype)
        for i in range(self.n_layers):
            p = f"dec.{i}"
            q, k, v = self._qkv(self._ln(x, f"{p}.norm_self"), f"{p}.self", self.n_heads, cos, sin)
            self.k_cache[i, :, :V] = k
            self.v_cache[i, :, :V] = v
            x = x + self._attend(q, k, v, w[f"{p}.self_o.weight"])
            x = x + self._ffn(self._ln(x, f"{p}.norm_ffn"), p)
        self.h.copy_(self._decoder_step(self.soa_embed[None], self.soa_position))
        self.frame_index.zero_()
        self.seen.zero_()

    def _step(self) -> None:
        """Sample frame_index's frame from h into frame_out, then step the global decoder."""
        u = self.frame_in
        K = self.num_codebooks
        is_eoa, codes, _, _ = self._frame(
            self.h, u[K + 1] > 0, self.sampling_in, u[0], u[1 : K + 1], self.seen
        )
        # index_put_ with a device value: `= True` copies a CPU scalar, which capture rejects.
        self.seen.index_put_((self.codebooks, codes), self.codebooks >= 0)
        self.frame_out[:1].copy_(is_eoa.view(1))
        self.frame_out[1:].copy_(codes)
        self.h.copy_(self._advance(codes, self.frame_index))
        self.frame_index += 1

    @contextmanager
    def _on_stream(self):
        with torch.no_grad(), torch.cuda.stream(self.stream):
            yield

    def _run(self, name: str) -> None:
        graph = self._graphs.get(name)
        if graph is not None:
            graph.replay()
        else:
            getattr(self, f"_{name}")()

    def _put(self, name: str) -> None:
        getattr(self, name).copy_(self._io[name][0], non_blocking=True)

    def _start(self, text_ids, voice_emb, sampling: Sampling) -> None:
        """Load one segment's inputs and run _begin."""
        ids = np.asarray(text_ids, dtype=np.int64).reshape(-1)
        if len(ids) > MAX_TEXT_TOKENS:
            raise ValueError(f"text has more than {MAX_TEXT_TOKENS} tokens")
        # Earlier asynchronous copies from these host buffers must be done before rewriting
        # them; normally the last frame already waited, so this costs nothing.
        self.stream.synchronize()
        text = self._io["text_in"][1]
        text[0], text[1 : 1 + len(ids)], text[1 + len(ids) :] = len(ids), ids, 0
        self._io["voice_in"][1][:] = np.asarray(voice_emb, np.float32).reshape(
            self.n_voice_queries, self.d_model
        )
        self._io["sampling_in"][1][:] = self._sampling_values(sampling)
        with self._on_stream():
            for name in ("text_in", "voice_in", "sampling_in"):
                self._put(name)
            self._run("begin")

    def _next_frame(self, ctrl_u, audio_u, forbid_eoa: bool) -> np.ndarray:
        """One frame step; returns [<eoa>, codes (K)] on the host."""
        u = self._io["frame_in"][1]
        u[0], u[1:-1], u[-1] = ctrl_u.reshape(()), audio_u.reshape(-1), forbid_eoa
        out, out_np = self._io["frame_out"]
        with self._on_stream():
            self._put("frame_in")
            self._run("step")
            out.copy_(self.frame_out, non_blocking=True)
        self.stream.synchronize()
        return out_np.copy()

    def frames(self, text, voice_emb, sampling: Sampling, rng: np.random.Generator):
        """ZeroTTS._generate_frames on this model: yields (1, K) int64 codes per frame."""
        if sampling.cfg_scale != 1:
            raise ValueError("runtime torch-cuda supports only cfg_scale = 1")
        self._start(self.tokenizer(text), voice_emb, sampling)
        t = 0
        tail_left = None  # None until <eoa>, then a countdown of tail frames to keep
        while True:
            ctrl_u, audio_u = draw_u(rng, self.num_codebooks)
            host = self._next_frame(ctrl_u, audio_u, t < MIN_FRAMES or tail_left is not None)
            if tail_left is None and host[0]:
                tail_left = EOA_EXTRA_FRAMES
            if (tail_left is not None and tail_left <= 0) or t >= MAX_FRAMES:
                break
            yield host[None, 1:]
            if tail_left is not None:
                tail_left -= 1
                if tail_left <= 0:
                    break
            t += 1

    def warmup(self) -> None:
        """Run both steps eagerly on the worker stream, capture them as graphs, replay once."""
        voice = np.zeros((self.n_voice_queries, self.d_model), dtype=np.float32)
        draws = draw_u(np.random.default_rng(0), self.num_codebooks)
        self._start(self.tokenizer("a"), voice, Sampling())
        self._next_frame(*draws, True)
        if self.cuda_graph and not self._graphs:
            pool = None
            for name in ("begin", "step"):
                graph = torch.cuda.CUDAGraph()
                with (
                    torch.no_grad(),
                    torch.cuda.graph(
                        graph, pool=pool, stream=self.stream, capture_error_mode="thread_local"
                    ),
                ):
                    getattr(self, f"_{name}")()
                pool = graph.pool()
                self._graphs[name] = graph
            self._start(self.tokenizer("a"), voice, Sampling())
            self._next_frame(*draws, True)
        torch.cuda.synchronize(self.device)

    # ── eager API for the parity tests (lockstep with ONNX, teacher forcing) ──
    # Each call waits for the caller's stream first (its seen/forced tensors) and syncs after.

    @contextmanager
    def _eager(self):
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with self._on_stream():
            yield
        self.stream.synchronize()

    def begin(self, text_ids: np.ndarray, voice_emb: np.ndarray) -> torch.Tensor:
        """Start a segment; returns the hidden for frame 0."""
        with self._eager():
            self._start(text_ids, voice_emb, Sampling())
            return self.h.clone()

    def frame(self, h, forbid_eoa: bool, sampling: Sampling, ctrl_u, audio_u, seen, forced=None):
        """_frame on host arguments, eagerly; see there."""
        with self._eager():
            params = torch.tensor(self._sampling_values(sampling), device=self.device)
            return self._frame(
                h,
                torch.tensor(bool(forbid_eoa), device=self.device),
                params,
                torch.as_tensor(ctrl_u, device=self.device).reshape(()),
                torch.as_tensor(audio_u, device=self.device).reshape(-1),
                seen,
                forced,
            )

    def advance(self, codes, frame_index: int) -> torch.Tensor:
        """Feed frame codes (K,) at frame_index's position, eagerly; returns the next hidden."""
        with self._eager():
            return self._advance(codes, torch.tensor([frame_index], device=self.device))


class TorchCodec:
    """The codec on one GPU, fp32: (B, K, T) codes -> (B, T_audio) float32 mono at 48 kHz.

    Same interface as model.codec.MossCodecDecoder (the ONNX codec): `decode` and
    `streaming_decoder`. Runs on the worker's stream. Built from the vendored modeling code and
    local files only; no transformers `from_pretrained`, no remote code, no download.

    Offline decoding runs eagerly. Streaming uses a second instance kept in streaming mode for
    the worker's lifetime: its causal state is device tensors updated in place, so each chunk
    size the engine uses (1..STREAM_MAX_CHUNK_FRAMES frames) replays as a CUDA Graph, and a new
    stream resets the state in place. A codec call is otherwise ~20 ms of Python launches.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: torch.device,
        stream: torch.cuda.Stream,
        cuda_graph=True,
    ):
        model_dir = Path(model_dir)
        # cuDNN runs fp32 convolutions as TF32 by default, which moves the PCM by ~1e-3.
        torch.backends.cudnn.allow_tf32 = False
        self.device, self.stream = device, stream
        config = json.loads((model_dir / "codec_config.json").read_text())
        self.num_codebooks = int(config["quantizer_kwargs"]["num_quantizers"])
        self.model = self._load(model_dir, config)
        # Load-time CUDA setup, and fail at load rather than on the first request.
        self.decode(np.zeros((1, self.num_codebooks, 1), dtype=np.int64))

        self._streaming = self._load(model_dir, config)
        self._stack = ExitStack()
        self._stack.enter_context(self._streaming.streaming(batch_size=1))
        self._states = [
            m._streaming_state
            for m in self._streaming.modules()
            if getattr(m, "_streaming_state", None) is not None
        ]
        self._reset_mask = torch.ones(1, dtype=torch.bool, device=device)
        self._graphs: dict[int, tuple] = {}
        sizes = range(1, STREAM_MAX_CHUNK_FRAMES + 1)
        with torch.no_grad(), torch.cuda.stream(stream):
            self._inputs = {
                n: torch.zeros(self.num_codebooks, 1, n, dtype=torch.int64, device=device)
                for n in sizes
            }
            for n in sizes:  # eager runs before capture
                self._decode_streaming(self._inputs[n])
            self.reset()
        if cuda_graph:
            pool = None
            for n in sizes:
                graph = torch.cuda.CUDAGraph()
                with (
                    torch.no_grad(),
                    torch.cuda.graph(
                        graph, pool=pool, stream=stream, capture_error_mode="thread_local"
                    ),
                ):
                    out = self._decode_streaming(self._inputs[n])
                pool = graph.pool()
                self._graphs[n] = (graph, *out)
        torch.cuda.synchronize(device)

    def _load(self, model_dir: Path, config: dict):
        from zerotts_server.model.moss.configuration_moss_audio_tokenizer import (
            MossAudioTokenizerConfig,
        )
        from zerotts_server.model.moss.modeling_moss_audio_tokenizer import (
            MossAudioTokenizerModel,
        )

        model = MossAudioTokenizerModel(MossAudioTokenizerConfig(**config))
        model.encoder = torch.nn.ModuleList()  # decode only; codec.safetensors has no encoder
        model.load_state_dict(load_file(model_dir / "codec.safetensors"), strict=True)
        return model.eval().to(self.device)

    @staticmethod
    def _mono(model, codes_kbt):
        """What the model's decode() and chunked batch_decode() call, on (K, B, T) codes; the
        codec is stereo and mono is the channel mean, as the ONNX codec."""
        out = model._decode_frame(codes_kbt)
        return out.audio.mean(dim=1), out.audio_lengths

    def _decode_streaming(self, codes_kbt):
        return self._mono(self._streaming, codes_kbt)

    def reset(self) -> None:
        """Back to a fresh streaming state, in place (the graphs keep their addresses)."""
        for state in self._states:
            state.reset(self._reset_mask)

    @staticmethod
    def _host(audio, lengths) -> np.ndarray:
        return audio[:, : int(lengths[0])].cpu().numpy()

    @torch.no_grad()
    def decode(self, codes_bkt: np.ndarray) -> np.ndarray:
        with torch.cuda.stream(self.stream):
            codes = torch.as_tensor(np.asarray(codes_bkt, dtype=np.int64), device=self.device)
            if codes.ndim == 2:  # (K, T) -> (1, K, T)
                codes = codes[None]
            return self._host(*self._mono(self.model, codes.transpose(0, 1)))

    @torch.no_grad()
    def decode_chunk(self, codes_bkt: np.ndarray) -> np.ndarray:
        """Streaming decode of a (1, K, n) chunk, n in 1..STREAM_MAX_CHUNK_FRAMES."""
        codes = np.asarray(codes_bkt, dtype=np.int64)
        n = codes.shape[-1]
        if n not in self._inputs:
            raise ValueError(f"streaming chunks are 1..{STREAM_MAX_CHUNK_FRAMES} frames")
        with torch.cuda.stream(self.stream):
            buffer = self._inputs[n]
            buffer.copy_(torch.from_numpy(codes.reshape(self.num_codebooks, 1, n)))
            if n in self._graphs:
                graph, audio, lengths = self._graphs[n]
                graph.replay()
            else:
                audio, lengths = self._decode_streaming(buffer)
            return self._host(audio, lengths)

    def streaming_decoder(self) -> "TorchStreamingDecoder":
        return TorchStreamingDecoder(self)


class TorchStreamingDecoder:
    """One stream's decoder: starts from a fresh codec state, kept across its chunks."""

    def __init__(self, codec: TorchCodec):
        self._codec = codec
        with torch.cuda.stream(codec.stream):
            codec.reset()

    def decode_chunk(self, codes_bkt: np.ndarray) -> np.ndarray:
        return self._codec.decode_chunk(codes_bkt)

    def close(self) -> None:
        pass
