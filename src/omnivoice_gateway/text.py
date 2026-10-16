"""Text pipeline: input text -> engine text. Pure functions, no I/O.

Language-independent steps only (spec steps 1-3, 8-9); the Vietnamese steps (4-7) plug in
between cleanup and finishing.
"""

import re
import unicodedata

# Bump whenever a rule changes so a mis-reading can be traced to the version that produced it.
RULES_VERSION = "1"
PIPELINE_VERSION = RULES_VERSION


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
_EM = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])|(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])")
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

# --- finishing ---------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?…][\"'”’»)\]]*$")
_CLAUSE_END = re.compile(r"[,;:]$")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?…])")
_WS = re.compile(r"\s+")


def process(text: str, *, max_chars: int, vietnamese: bool) -> tuple[str, list[str]]:
    """Return (engine text, text warnings in detection order) or raise TextError.

    `vietnamese` is true when the effective language is Vietnamese and normalization is on.
    """
    text = unicodedata.normalize("NFC", text)
    if len(text) > max_chars:
        raise TextError("text_too_long")
    if _THINK.search(text):
        raise TextError("reasoning_markup")
    if _CONTROL.search(text):
        raise TextError("model_control")

    text, warnings = _clean(text)
    # Vietnamese steps (tag protection, lexicon, sea-g2p, <en> handling) are not implemented yet.
    text = _finish(text)
    if not any(ch.isalnum() for ch in text):
        raise TextError("empty_text")
    return text, warnings


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


def _ends_clause(text: str) -> bool:
    return bool(_SENTENCE_END.search(text) or _CLAUSE_END.search(text))


def _finish(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", _WS.sub(" ", text)).strip()
    if _SENTENCE_END.search(text):
        return text
    return _CLAUSE_END.sub("", text) + "."
