"""Text pipeline: input text -> engine text.

Spec step order: NFC and length (1), guard (2), cleanup (3), then for Vietnamese with
normalization on: tag protection (4), pronunciation lexicon (5), sea-g2p (6), <en> handling (7);
finally finishing (8) and the empty check (9). Only the lexicon is loaded from disk, at startup.
"""

import csv
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path

from sea_g2p import Normalizer

# Bump whenever a rule changes so a mis-reading can be traced to the version that produced it.
# The pipeline version also carries a hash of the lexicon (see Pipeline.version).
RULES_VERSION = "2"

LEXICON_FIELDS = ("word", "replacement", "note")


class TextError(Exception):
    """An input error; `type` is the public error code, the message never contains request text."""

    MESSAGES = {
        "text_too_long": "input text exceeds the maximum length",
        "reasoning_markup": "input text contains reasoning markup (think tags)",
        "model_control": "input text contains model control tokens",
        "empty_text": "input text has nothing left to speak",
    }

    def __init__(self, type_: str):
        super().__init__(self.MESSAGES[type_])
        self.type = type_


def is_vietnamese(language: str | None) -> bool:
    if language is None:
        return False
    language = language.lower()
    return language in ("vi", "vietnamese") or language.startswith(("vi-", "vi_"))


# --- guard -------------------------------------------------------------------

# Exact tag name (not <thinking>); an unfinished "<think" at the end also counts, failing closed.
_THINK = re.compile(r"</?\s*think(?=[\s/>]|$)", re.IGNORECASE)
# OmniVoice and Qwen control tokens: <|im_start|>, <|denoise|>, <|lang_start|>, ...
_CONTROL = re.compile(r"<\|[^<>|\n]*\|>")

# --- cleanup -----------------------------------------------------------------

# A fence runs to a matching closing fence, or to the end of the text when left open.
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?$.*?(?:^[ \t]*\1[ \t]*$|\Z)", re.MULTILINE | re.DOTALL)
# A tag starts with a letter right after "<" or "</", so comparisons like "a < b" survive.
_HTML = re.compile(r"<!--.*?(?:-->|\Z)|</?[A-Za-z][\w:.-]*(?:\s[^<>]*)?/?>", re.DOTALL)
_LINK = re.compile(r"!?\[([^\]\n]*)\]\([^)\s]*\)")
# Stop before trailing sentence punctuation so "... tại https://x.vn." keeps its period.
# Not preceded by "@" or a word char, so emails (kept for the normalizer) are not cut.
_URL = re.compile(r"<?(?<![\w@.])(?:https?://|www\.)[^\s<>()\[\]]*[^\s<>()\[\].,;:!?'\"]>?")
_RULE = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.MULTILINE)
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
_QUOTE = re.compile(r"^[ \t]*(?:>[ \t]?)+", re.MULTILINE)
_BULLET = re.compile(r"^[ \t]*(?:[-*+•]|\d{1,9}[.)])[ \t]+", re.MULTILINE)
_STRONG = re.compile(r"(\*\*|__|~~)(?!\s)(.+?)(?<!\s)\1")
# Single delimiters only around words, so "2*3" and snake_case survive.
_EM = re.compile(
    r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])|(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])"
)
# Unpaired leftovers (e.g. bold spanning lines) that the normalizer would read as symbols.
_STRAY = re.compile(r"\*{2,}|~{2,}")
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, transport, flags, skin tones
    "\u2600-\u27bf"  # misc symbols, dingbats
    "\u2b00-\u2bff"  # arrows and stars
    "\u231a\u231b\u23e9-\u23f3\u23f8-\u23fa"  # watch, hourglass, media controls
    "\ufe0e\ufe0f\u200d\u20e3"  # variation selectors, ZWJ, keycap
    "]+"
)

# --- Vietnamese ---------------------------------------------------------------

# The 13 OmniVoice non-verbal tags (same list as audio.cpp's tokenizer).
_NONVERBAL = re.compile(
    r"\[(?:laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi"
    r"|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\]"
)
# CMU pronunciation override: uppercase ARPAbet phones, vowels optionally with a stress digit.
_PHONE = (
    r"(?:AA|AE|AH|AO|AW|AY|B|CH|DH|D|EH|ER|EY|F|G|HH|IH|IY|JH|K|L|M|NG|N"
    r"|OW|OY|P|R|SH|S|TH|T|UH|UW|V|W|Y|ZH|Z)[0-2]?"
)
# One capture group so re.split alternates text segments and protected spans.
_PROTECTED = re.compile(rf"({_NONVERBAL.pattern}|\[{_PHONE}(?: {_PHONE})*\])")
_LEADING_PUNCT = re.compile(r"[\s,.;:!?…]*")
_EN = re.compile(r"<en>(.*?)</en>", re.DOTALL)
# sea-g2p spells acronyms as spaced lowercase letters inside <en> ("AI" -> "<en>a i</en>").
_LETTER_RUN = re.compile(r"(?<!\S)[a-z](?: [a-z])+(?!\S)")

# One shared instance; loading its dictionary is the expensive part.
_normalizer = Normalizer("vi")

# --- finishing ---------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?…][\"'”’»)\]]*$")
_CLAUSE_END = re.compile(r"[,;:]$")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?…])")
_WS = re.compile(r"\s+")


def load_lexicon(path: str | Path) -> dict[str, str]:
    """Read the pronunciation lexicon: a CSV with header word,replacement,note."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != LEXICON_FIELDS:
            raise ValueError(f"{path}: header must be {','.join(LEXICON_FIELDS)}")
        lexicon = {}
        for row in reader:
            word, replacement = (
                unicodedata.normalize("NFC", (row[key] or "").strip()) for key in LEXICON_FIELDS[:2]
            )
            if not word and not replacement:
                continue
            if not word or not replacement:
                raise ValueError(f"{path}:{reader.line_num}: word and replacement are required")
            if word in lexicon:
                raise ValueError(f"{path}:{reader.line_num}: duplicate word")
            lexicon[word] = replacement
    return lexicon


class Pipeline:
    def __init__(self, lexicon: Mapping[str, str] | None = None, *, max_chars: int):
        self.max_chars = max_chars
        lexicon = dict(lexicon or {})
        # Longest entry first so it wins over a shorter one at the same position; the lookarounds
        # keep matches on word boundaries so an entry never fires inside a longer word.
        words = sorted(lexicon, key=len, reverse=True)
        self._lexicon = lexicon
        self._lexicon_re = (
            re.compile(rf"(?<!\w)(?:{'|'.join(map(re.escape, words))})(?!\w)") if words else None
        )
        digest = hashlib.sha256(
            json.dumps(sorted(lexicon.items()), ensure_ascii=False).encode()
        ).hexdigest()
        self.version = f"{RULES_VERSION}-{digest[:8]}"

    def process(self, text: str, *, vietnamese: bool) -> tuple[str, list[str]]:
        """Return (engine text, text warnings in detection order) or raise TextError.

        `vietnamese` is true when the effective language is Vietnamese and normalization is on.
        """
        text = unicodedata.normalize("NFC", text)
        if len(text) > self.max_chars:
            raise TextError("text_too_long")
        if _THINK.search(text):
            raise TextError("reasoning_markup")
        if _CONTROL.search(text):
            raise TextError("model_control")

        text, warnings = _clean(text)
        if vietnamese:
            text = self._vietnamese(text)
        text = _finish(text)
        if not any(ch.isalnum() for ch in _NONVERBAL.sub(" ", text)):
            raise TextError("empty_text")
        return text, warnings

    def _vietnamese(self, text: str) -> str:
        # sea-g2p deletes tags ("[laughter]" -> "laughter,") and reads CMU phones as letters, so
        # protected spans are held out and only the text between them is normalized.
        parts = _PROTECTED.split(text)
        segments = [part.replace("[", " ").replace("]", " ") for part in parts[::2]]
        if self._lexicon_re:
            segments = [
                self._lexicon_re.sub(lambda m: self._lexicon[m.group()], seg) for seg in segments
            ]
        # sea-g2p drops a comma that starts its input, so punctuation right after a tag stays out.
        leads = [_LEADING_PUNCT.match(seg).group() for seg in segments]
        segments = [seg[len(lead) :] for seg, lead in zip(segments, leads, strict=True)]
        spoken = iter(
            _normalizer.normalize_batch([seg for seg in segments if seg.strip()], punc_norm=False)
        )
        segments = [
            lead + (_unwrap_en(next(spoken)) if seg.strip() else "")
            for seg, lead in zip(segments, leads, strict=True)
        ]
        parts[::2] = segments
        return " ".join(parts)


def _clean(text: str) -> tuple[str, list[str]]:
    warnings = []

    def drop(pattern: re.Pattern, warning: str, repl: str = " ") -> None:
        nonlocal text
        text, count = pattern.subn(repl, text)
        if count:
            warnings.append(warning)

    # Fences first so code content is never treated as markup or prose.
    drop(_FENCE, "omitted_code_block", "\n")
    text = text.replace("`", "")  # inline code keeps its content
    drop(_HTML, "omitted_html")
    text = _LINK.sub(r"\1", text)
    drop(_URL, "omitted_url")
    for pattern in (_RULE, _HEADING, _QUOTE, _BULLET):
        text = pattern.sub("", text)
    text = _STRONG.sub(r"\2", text)
    text = _EM.sub(lambda m: m.group(1) or m.group(2), text)
    text = _STRAY.sub(" ", text)
    drop(_EMOJI, "omitted_emoji")

    # A line break in markup (heading, list item, paragraph) ends a sentence; keep that boundary
    # so the upstream's chunker and prosody do not run lines together.
    lines = [_WS.sub(" ", line).strip() for line in text.split("\n")]
    text = " ".join(line if _ends_clause(line) else line + "." for line in lines if line)
    return text, warnings


def _unwrap_en(text: str) -> str:
    # Spelled letters become one uppercase acronym so the reading stays consistent; final
    # reading policy is decided by the listening run, the lexicon overrides per word.
    return _EN.sub(
        lambda m: _LETTER_RUN.sub(lambda r: r.group().replace(" ", "").upper(), m.group(1)), text
    )


def _ends_clause(text: str) -> bool:
    return bool(_SENTENCE_END.search(text) or _CLAUSE_END.search(text))


def _finish(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", _WS.sub(" ", text)).strip()
    if _SENTENCE_END.search(text):
        return text
    return _CLAUSE_END.sub("", text) + "."
