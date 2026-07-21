from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter

from fraude_detector.pdf_revisions import find_valid_revision_end_offsets


def test_detects_only_retained_incremental_updates(vector_pdf: Path, tmp_path: Path) -> None:
    initial = vector_pdf.read_bytes()
    assert len(find_valid_revision_end_offsets(initial)) == 1

    incremented = tmp_path / "incremented.pdf"
    writer = PdfWriter(vector_pdf, incremental=True)
    writer.add_metadata({"/ModDate": "D:20260721130000+02'00'"})
    writer.write(incremented)

    offsets = find_valid_revision_end_offsets(incremented.read_bytes())
    assert len(offsets) == 2
    assert offsets[0] < offsets[1]


def test_marker_like_text_is_not_a_revision() -> None:
    fake = b"%PDF-1.7\nstream\n%%EOF\nendstream\n"
    assert find_valid_revision_end_offsets(fake) == ()
