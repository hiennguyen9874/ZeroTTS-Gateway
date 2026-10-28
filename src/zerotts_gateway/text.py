"""Text pipeline: input text -> engine text.

Spec step order: NFC and length (1), guard (2), cleanup incl. markdown tables (3), then for
Vietnamese with normalization on: tag protection (4), pronunciation lexicon (5), reading rules
(6, ADR 0003), sea-g2p (7), <en> handling (8); finally finishing (9) and the empty check (10).
Only the lexicon is loaded from disk, at startup.
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
RULES_VERSION = "7"

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
# Markdown table: header row, delimiter row (`|---|:---:|`), then body rows; each row has a pipe.
_TABLE_DELIMITER = re.compile(r"^[ \t]*\|?(?:[ \t]*:?-+:?[ \t]*\|)+(?:[ \t]*:?-+:?[ \t]*)?$")
_TABLE_CELL_SPLIT = re.compile(r"(?<!\\)\|")
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
# A tag with the spaces around it and the punctuation right after it, for models without tags.
_TAG_SPAN = re.compile(rf"\s*{_PROTECTED.pattern}\s*(?P<punct>[,.;:!?…]*)")
_PUNCT_END = ",.;:!?…"
_CLOSING = ")]}\"'”’»"
_EN = re.compile(r"<en>(.*?)</en>", re.DOTALL)
# sea-g2p spells acronyms as spaced lowercase letters inside <en> ("AI" -> "<en>a i</en>").
_LETTER_RUN = re.compile(r"(?<!\S)[a-z](?: [a-z])+(?!\S)")

# One shared instance; loading its dictionary is the expensive part.
_normalizer = Normalizer("vi")

# --- reading rules (ADR 0003) -------------------------------------------------

# A version's parts: "2.0", "3.12", "10.4.1". Not followed by more digits, a word or a percent sign.
_VERSION = r"(\d+(?:\.\d+)+)(?![\w%]|[.,]\d)"
# "v2.0", "v10.4.1", "v2": the prefix is a version marker. sea-g2p reads "v2.0" as "vê hai. không".
_V_VERSION = re.compile(r"(?<![\w.])[vV](\d+(?:\.\d+)*)(?![\w%]|[.,]\d)")
# A name right before a version: an ASCII word with an uppercase letter ("Python", "GPT", "iOS"),
# joined by a space or a hyphen. sea-g2p reads "GPT-3.5" as "gờ phê tê gạch nối ba chấm năm".
_NAME_VERSION = re.compile(rf"(?<![\w.-])([A-Za-z]*[A-Z][A-Za-z]*)(?:-| )(?:{_VERSION})")
# Dot-grouped thousands ("USD 1.000.000") are an amount, not a version.
_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+")
# Title abbreviations with a period before a name. sea-g2p expands "ThS." but keeps the period
# ("thạc sĩ. nam"), which is a sentence break before the name.
_TITLES = {
    "GS": "giáo sư",
    "PGS": "phó giáo sư",
    "TS": "tiến sĩ",
    "TSKH": "tiến sĩ khoa học",
    "ThS": "thạc sĩ",
    "Th.S": "thạc sĩ",
    "BS": "bác sĩ",
    "KS": "kỹ sư",
    "GV": "giáo viên",
}
_TITLE = "|".join(map(re.escape, sorted(_TITLES, key=len, reverse=True)))
# A chain of titles, each with its period ("PGS.TS.", "ThS. BS."), right before a letter.
_TITLE_CHAIN = re.compile(rf"(?<![\w.])(?:(?:{_TITLE})\. ?)+(?=[^\W\d_])")
_TITLE_ONE = re.compile(rf"({_TITLE})\.")

_DIGIT_WORDS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
# A number that is not part of a longer token: no word char, dot, comma or sign right before it.
_NUM_START = r"(?<![\w.,+/:-])"
# ... and nothing numeric or wordy right after it ("1.250.000", "12/2024", "50kg").
_NUM_END = r"(?!\w|[.,/:]\d)"
# Hotline 1800/1900 with its number, grouped or not ("1900 1234", "1800.6868", "19001234").
# sea-g2p reads the second group as a cardinal ("một nghìn hai trăm ba mươi bốn").
_HOTLINE = re.compile(rf"{_NUM_START}(1[89]00)((?:[ .-]?\d{{2,4}}){{1,3}}){_NUM_END}")
_DIGIT_GROUP = re.compile(r"\d+")
# A digit string right after a code keyword ("Mã OTP là 123456", "STK 0123 4567"): read digit by
# digit. Uppercase acronyms are case-sensitive so "pin 5000mAh" (battery) is not a PIN.
_CODE_WORD = (
    r"(?:[Mm]ã(?: (?:đơn(?: hàng)?|số(?: thuế)?|giao dịch|xác nhận))?|OTP|PIN|STK|[Ss]ố tài khoản)"
)
_CODE = re.compile(
    rf"(?<!\w)({_CODE_WORD}(?:\s*[:#]|\s+là|\s+{_CODE_WORD})*\s*)"
    rf"(\d+(?:[ .-]\d{{2,}})*){_NUM_END}"
)
# A digit string of 4+ digits starting with 0 is a phone number or code, never an amount.
_LEADING_ZERO = re.compile(rf"{_NUM_START}0\d{{3,}}{_NUM_END}")
# Quarter with a year ("Q1/2024", "Q4-2023"); sea-g2p reads the year digit by digit. "Q1" alone is
# left as is: it is just as often "quận 1".
_QUARTER = re.compile(r"(?<!\w)Q([1-4]) ?[/-] ?(\d{4})(?!\w|[.,/]\d)")
# A time ("8h", "8h30", "8:30") or a bare hour before a dashed range ("8-17h"). sea-g2p drops the
# dash of a time range, so "đến" is missing.
_TIME = r"\d{1,2}(?:h(?:\d{2})?|:\d{2})"
_TIME_RANGE = re.compile(rf"{_NUM_START}({_TIME}|\d{{1,2}}) ?[-–] ?({_TIME})(?!\w|[.,:]\d)")
# A day range before a month ("25-26/12", "25-26/12/2024"); sea-g2p reads "/12" as "trên mười hai".
_DAY_RANGE = re.compile(
    rf"{_NUM_START}(\d{{1,2}}) ?[-–] ?(\d{{1,2}})/(\d{{1,2}})(?:/(\d{{4}}))?{_NUM_END}"
)
# A date word right before day/month: read as a date even when the date does not exist
# ("ngày 31/02"), sea-g2p then gives "ba mươi mốt trên không hai".
_DATE_WORD = r"(?:[Nn]gày|[Hh]ôm|[Mm]ùng|[Mm]ồng|[Ss]áng|[Tt]rưa|[Cc]hiều|[Tt]ối|[Đđ]êm|[Hh]ạn)"
_CONTEXT_DATE = re.compile(
    rf"(?<!\w)({_DATE_WORD}) (\d{{1,2}})/(\d{{1,2}})(?:/(\d{{4}}))?{_NUM_END}"
)
# Without a date word, a/b with a < b <= 10 is a fraction ("giảm 1/3 giá"); "7/3" stays.
_FRACTION = re.compile(
    rf"(?<!\w)(?:([Tt]háng|[Nn]ăm|[Qq]uý) )?{_NUM_START}(\d{{1,2}})/(\d{{1,2}}){_NUM_END}"
)
# Money slang right after the digits: "50k" (sea-g2p: "năm mươi ca"), "2tr", "2tr5".
_SLANG_K = re.compile(rf"{_NUM_START}(\d+(?:[.,]\d{{1,2}})?)[kK](?!\w)")
_SLANG_TR = re.compile(rf"{_NUM_START}(\d+)tr(\d{{1,3}})?(?!\w)")
# English-style decimal: one or two digits after a dot ("3.5"). Runs after the version rules, so
# "v2.0", "Python 3.12" and "GPT-3.5" keep "chấm"; three digits ("1.250") are thousands.
_DECIMAL = re.compile(r"(?<![\w.,])(?<!\w-)(\d+)\.(\d{1,2})(?!\d|[.,]\d)")
# A plain 7 to 9 digit amount ("1250000"); sea-g2p reads it digit by digit like a phone number.
_PLAIN_AMOUNT = re.compile(rf"{_NUM_START}[1-9]\d{{6,8}}{_NUM_END}")
# Metric units with a superscript: sea-g2p says "mét bình phương" for "m²" but "mét vuông" for "m2".
_UNIT_PREFIX = {"": "", "k": "ki lô ", "h": "héc tô ", "d": "đề xi ", "c": "xen ti ", "m": "mi li "}
_SUPERSCRIPT_UNIT = re.compile(r"(?<![^\W\d_])(k|h|d|c|m)?m([²³])(?![^\W\d_])")

# --- finishing ---------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?…][\"'”’»)\]]*$")
_CLAUSE_END = re.compile(r"[,;:]$")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?…])")
_WS = re.compile(r"\s+")
# A markup line (list item, heading, paragraph line) of at most this many syllables joins the next
# with a comma instead of a period (issue #15).
SHORT_LINE_SYLLABLES = 4


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

    def process(
        self, text: str, *, vietnamese: bool, nonverbal_tags: bool = True
    ) -> tuple[str, list[str]]:
        """Return (engine text, text warnings in detection order) or raise TextError.

        `vietnamese` is true when the effective language is Vietnamese and normalization is on.
        `nonverbal_tags` is false for a model that would read tags and CMU phones as words.
        """
        text = unicodedata.normalize("NFC", text)
        if len(text) > self.max_chars:
            raise TextError("text_too_long")
        if _THINK.search(text):
            raise TextError("reasoning_markup")
        if _CONTROL.search(text):
            raise TextError("model_control")

        text, warnings = _clean(text)
        if not nonverbal_tags:
            text, count = _drop_tags(text)
            if count:
                warnings.append("omitted_nonverbal_tag")
        if vietnamese:
            text = self._vietnamese(text)
        text = _finish(text)
        if not any(ch.isalnum() for ch in _NONVERBAL.sub(" ", text)):
            raise TextError("empty_text")
        return text, warnings

    def _apply_lexicon(self, text: str) -> str:
        # Reserve all literal matches first: even a separate numeric entry must win over a
        # reading rule. Keep the original name's version context before replacing it with
        # spoken Vietnamese ("iOS 18.2" -> "ai ô ét mười tám chấm hai", not "phẩy").
        matches = list(self._lexicon_re.finditer(text))
        out = []
        end = 0
        for i, match in enumerate(matches):
            out.extend((text[end : match.start()], self._lexicon[match.group()]))
            end = match.end()
            version = _NAME_VERSION.match(text, match.start())
            next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            if version and version.end(1) == end and version.end() <= next_start:
                # Reuse the version rule, including its thousands exception; a literal entry
                # for the whole form ("GPT-3.5") does not take this branch.
                out.append(_name_version(version)[len(match.group()) :])
                end = version.end()
        out.append(text[end:])
        return "".join(out)

    def _vietnamese(self, text: str) -> str:
        # sea-g2p deletes tags ("[laughter]" -> "laughter,") and reads CMU phones as letters, so
        # protected spans are held out and only the text between them is normalized.
        parts = _PROTECTED.split(text)
        segments = [part.replace("[", " ").replace("]", " ") for part in parts[::2]]
        if self._lexicon_re:
            segments = [self._apply_lexicon(seg) for seg in segments]
        # After the lexicon, so an entry for a written form ("GPT-3.5") wins over the rule.
        segments = [_reading_rules(seg) for seg in segments]
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
    text = _tables(text)
    for pattern in (_RULE, _HEADING, _QUOTE, _BULLET):
        text = pattern.sub("", text)
    text = _STRONG.sub(r"\2", text)
    text = _EM.sub(lambda m: m.group(1) or m.group(2), text)
    text = _STRAY.sub(" ", text)
    drop(_EMOJI, "omitted_emoji")

    # A line break in markup (heading, list item, paragraph) is a boundary; keep it so the
    # upstream's chunker and prosody do not run lines together. A short line without its own
    # punctuation ends in a comma, not a period: audio.cpp sizes the audio by character weight
    # with no time for sentence pauses, so runs of very short sentences lose a whole sentence.
    # Its chunker also splits at commas, so chunk boundaries survive (ADR 0001).
    lines = [line for line in (_WS.sub(" ", line).strip() for line in text.split("\n")) if line]
    text = " ".join(_end_line(line, last=i == len(lines) - 1) for i, line in enumerate(lines))
    return text, warnings


def _drop_tags(text: str) -> tuple[str, int]:
    """Remove non-verbal tags and CMU phones; return (text, number removed).

    Punctuation right after a tag is kept only when it does not follow other punctuation or the
    start of the text, so "A [laughter], B" -> "A, B" and "[sigh]. A" -> "A".
    """
    out = ""
    end = count = 0
    for m in _TAG_SPAN.finditer(text):
        out = (out + text[end : m.start()]).rstrip()
        punct = m.group("punct")
        if not out or out[-1] in _PUNCT_END:
            punct = ""
        end = m.end()
        # No space before a closing bracket or quote: "(cười [laughter])" -> "(cười)".
        out += punct + ("" if text[end : end + 1] in _CLOSING and end < len(text) else " ")
        count += 1
    return out + text[end:], count


def _tables(text: str) -> str:
    """Drop table delimiter rows; each row becomes one line of comma-joined cells.

    The line then ends like any markup line, so short rows join with commas (issue #15).
    """
    lines = text.split("\n")
    out = []
    in_table = False
    for i, line in enumerate(lines):
        if _TABLE_DELIMITER.match(line) and out and "|" in lines[i - 1]:
            out[-1] = _table_row(lines[i - 1])
            in_table = True
            continue
        if in_table and "|" in line and line.strip():
            out.append(_table_row(line))
            continue
        in_table = False
        out.append(line)
    return "\n".join(out)


def _table_row(line: str) -> str:
    cells = [cell.replace("\\|", "|").strip() for cell in _TABLE_CELL_SPLIT.split(line)]
    cells = [cell for cell in cells if cell]
    row = ""
    for i, cell in enumerate(cells):
        if row:
            row += " " if _SENTENCE_END.search(row) else ", "
        # A cell's own trailing comma or colon would double the joining comma.
        row += cell if i == len(cells) - 1 else _CLAUSE_END.sub("", cell)
    return row


def _reading_rules(text: str) -> str:
    """Rewrite written forms sea-g2p mis-reads into spoken Vietnamese it leaves as is."""
    text = _TITLE_CHAIN.sub(_title_chain, text)
    text = _V_VERSION.sub(lambda m: "vê " + _version_words(m.group(1)), text)
    text = _NAME_VERSION.sub(_name_version, text)
    # Digit-by-digit forms first, so no number rule below reads a code as an amount or date.
    text = _HOTLINE.sub(lambda m: _digit_groups(f"{m.group(1)} {m.group(2)}"), text)
    text = _CODE.sub(lambda m: m.group(1) + _digit_groups(m.group(2)), text)
    text = _LEADING_ZERO.sub(lambda m: _digit_groups(m.group()), text)
    text = _QUARTER.sub(lambda m: f"quý {m.group(1)} năm {m.group(2)}", text)
    text = _TIME_RANGE.sub(_time_range, text)
    text = _DAY_RANGE.sub(_day_range, text)
    text = _CONTEXT_DATE.sub(_context_date, text)
    text = _FRACTION.sub(_fraction, text)
    text = _SLANG_K.sub(lambda m: f"{m.group(1)} nghìn", text)
    text = _SLANG_TR.sub(_slang_tr, text)
    text = _DECIMAL.sub(r"\1,\2", text)
    text = _PLAIN_AMOUNT.sub(lambda m: f"{int(m.group()):,}".replace(",", "."), text)
    return _SUPERSCRIPT_UNIT.sub(_superscript_unit, text)


def _digit_groups(digits: str) -> str:
    # Words, so sea-g2p cannot read a group as a cardinal; groups keep a pause like "+84 912 345".
    return ", ".join(
        " ".join(_DIGIT_WORDS[int(d)] for d in group) for group in _DIGIT_GROUP.findall(digits)
    )


def _time_range(m: re.Match) -> str:
    return f"{m.group(1)} đến {m.group(2)}"


def _day_month(day: str, month: str, year: str | None) -> str | None:
    if not (1 <= int(day) <= 31 and 1 <= int(month) <= 12):
        return None
    return f"{int(day)} tháng {int(month)}" + (f" năm {year}" if year else "")


def _day_range(m: re.Match) -> str:
    first, day, month, year = m.groups()
    rest = _day_month(day, month, year)
    if rest is None or not 1 <= int(first) < int(day):
        return m.group()
    return f"{int(first)} đến {rest}"


def _context_date(m: re.Match) -> str:
    word, day, month, year = m.groups()
    date = _day_month(day, month, year)
    return m.group() if date is None else f"{word} {date}"


def _fraction(m: re.Match) -> str:
    word, a, b = m.groups()
    # "tháng 5/12", "năm 1/2" belong to sea-g2p's date reading; only a bare a/b is a fraction.
    if word or not 0 < int(a) < int(b) <= 10:
        return m.group()
    return f"{int(a)} phần {int(b)}"


def _slang_tr(m: re.Match) -> str:
    millions, rest = m.groups()
    if not rest:
        return f"{millions} triệu"
    # "2tr5" is 2.5 million: the digits after "tr" are the leading digits of the thousands.
    return f"{millions} triệu {int(rest.ljust(3, '0'))} nghìn"


def _superscript_unit(m: re.Match) -> str:
    prefix, power = m.groups()
    unit = _UNIT_PREFIX[prefix or ""] + "mét " + ("vuông" if power == "²" else "khối")
    # "100m²" -> "100 mét vuông": the space keeps the number and the unit apart.
    before = m.string[m.start() - 1 : m.start()]
    return (" " if before.isdigit() else "") + unit


def _title_chain(m: re.Match) -> str:
    rest = m.string[m.end() :]
    # Only before a capitalized name; "... gặp ThS. hôm qua" may be a sentence end.
    if not rest[:1].isupper():
        return m.group()
    return " ".join(_TITLES[t] for t in _TITLE_ONE.findall(m.group())) + " "


def _name_version(m: re.Match) -> str:
    if _THOUSANDS.fullmatch(m.group(2)):
        return m.group()
    return f"{m.group(1)} {_version_words(m.group(2))}"


def _version_words(version: str) -> str:
    # Each part is a cardinal number ("3.12" -> "ba chấm mười hai"); sea-g2p reads "05" digit by
    # digit, which suits a zero-padded part.
    return " chấm ".join(
        _normalizer.normalize(part, punc_norm=False) for part in version.split(".")
    )


def _end_line(line: str, *, last: bool) -> str:
    short = not last and (
        sum(1 for word in line.split() if any(ch.isalnum() for ch in word)) <= SHORT_LINE_SYLLABLES
    )
    # A short lead-in like "Cần mang:" before a list is dropped as often as a short sentence;
    # a comma there is heard far more reliably (issue #15).
    if short and line.endswith(":"):
        return line[:-1] + ","
    if _ends_clause(line):
        return line
    return line + ("," if short else ".")


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
