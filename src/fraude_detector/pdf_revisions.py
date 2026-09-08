"""Validated discovery of PDF incremental revisions.

The number of valid revisions is not a save counter: a full rewrite removes the
old history. We expose only incremental updates that still exist in the file.
"""

from __future__ import annotations

import io
import re
from contextlib import suppress

import pypdfium2
from pypdf import PdfReader

from fraude_detector.pdf_logging import suppress_pypdf_recovery_messages

_EOF_PATTERN = re.compile(rb"%%EOF(?=[\x00\x09\x0a\x0c\x0d\x20]|$)")
_STARTXREF_AT_END = re.compile(rb"startxref\s+(\d+)\s+%%EOF\s*$", re.DOTALL)


def find_valid_revision_end_offsets(
    raw_pdf: bytes,
    password: str | None = None,
) -> tuple[int, ...]:
    """Return byte offsets ending each independently readable PDF revision.

    Candidate ``%%EOF`` markers are accepted only when they have a coherent
    ``startxref`` pointer and both pypdf (strict mode) and PDFium can open the
    prefix. This avoids treating marker-like text inside streams as revisions.
    """

    valid: list[int] = []
    for eof_match in _EOF_PATTERN.finditer(raw_pdf):
        end_offset = eof_match.end()
        candidate = raw_pdf[:end_offset]
        if not _has_valid_startxref(candidate):
            continue
        if _is_readable_pdf(candidate, password=password):
            valid.append(end_offset)

    return tuple(dict.fromkeys(valid))


def _has_valid_startxref(candidate: bytes) -> bool:
    match = _STARTXREF_AT_END.search(candidate)
    if match is None:
        return False

    xref_offset = int(match.group(1))
    if xref_offset < 0 or xref_offset >= len(candidate):
        return False

    target = candidate[xref_offset : xref_offset + 128].lstrip()
    if target.startswith(b"xref"):
        return True

    # Cross-reference streams start at an indirect object rather than at the
    # literal ``xref`` keyword.
    return bool(re.match(rb"\d+\s+\d+\s+obj\b", target))


def _is_readable_pdf(candidate: bytes, password: str | None) -> bool:
    try:
        with suppress_pypdf_recovery_messages():
            reader = PdfReader(io.BytesIO(candidate), strict=True)
            if reader.is_encrypted and (not password or reader.decrypt(password) == 0):
                return False
            if len(reader.pages) < 1:
                return False
    except Exception:
        return False

    pdfium_document: pypdfium2.PdfDocument | None = None
    try:
        pdfium_document = pypdfium2.PdfDocument(candidate, password=password)
        return len(pdfium_document) >= 1
    except Exception:
        return False
    finally:
        if pdfium_document is not None:
            with suppress(Exception):
                pdfium_document.close()
