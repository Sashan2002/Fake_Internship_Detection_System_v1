"""
Minimal, conservative text normalisation.

Per Section 8.1: for transformer models, avoid aggressive stemming or
stop-word removal because these operations can remove contextual
information. This module intentionally does *not* lowercase aggressively,
strip stopwords, or stem - it only removes structural noise (HTML tags,
excess whitespace, control characters) so the transformer sees language
that is as close to the original as possible.

This is used identically at training time and at inference time
(backend/utils/preprocessing.py imports from here) to avoid train/serve
skew.
"""
import html
import re
import unicodedata

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def strip_html(text: str) -> str:
    text = html.unescape(text or "")
    return _HTML_TAG_RE.sub(" ", text)


def normalise_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def collapse_whitespace(text: str) -> str:
    return _MULTI_WHITESPACE_RE.sub(" ", text or "").strip()


def remove_control_chars(text: str) -> str:
    return _CONTROL_CHARS_RE.sub("", text or "")


def minimal_normalise(text: str) -> str:
    """
    The single normalisation function used everywhere text reaches the
    tokenizer. Order matters: strip HTML/control chars before whitespace
    collapse; do NOT lowercase (BERT/RoBERTa cased models rely on case;
    if an uncased checkpoint is later selected, lowercasing should happen
    inside the tokenizer wrapper, not here, to keep this function reusable).
    """
    text = text or ""
    text = strip_html(text)
    text = normalise_unicode(text)
    text = remove_control_chars(text)
    text = collapse_whitespace(text)
    return text


def detect_url(text: str) -> bool:
    return bool(_URL_RE.search(text or ""))


def detect_email(text: str) -> bool:
    return bool(_EMAIL_RE.search(text or ""))


def detect_phone(text: str) -> bool:
    """
    Conservative phone-number heuristic: sequences of 7+ digits allowing
    common separators. Intentionally permissive-but-bounded to avoid
    false positives on ordinary numbers (e.g. postal codes) - this is a
    weak-supervision indicator, not a validator (see Section 9's
    interpretation rule: these are indicators, not proof).
    """
    phone_re = re.compile(r"(\+?\d[\d\-.\s()]{6,}\d)")
    for match in phone_re.finditer(text or ""):
        digits = re.sub(r"\D", "", match.group(0))
        if 7 <= len(digits) <= 15:
            return True
    return False
