from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pypdfium2
import pytest
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas

from fraude_detector.image_assets import decode_image_asset, extract_image_assets


def test_embedded_jpeg_is_extracted_without_recompression(tmp_path: Path) -> None:
    jpeg_path = tmp_path / "source.jpg"
    image = Image.new("RGB", (64, 48), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((3, 3, 60, 44), fill=(42, 91, 173), outline=(10, 20, 30), width=2)
    image.save(jpeg_path, format="JPEG", quality=83, subsampling=0)
    jpeg_bytes = jpeg_path.read_bytes()

    pdf_path = _image_pdf(tmp_path, jpeg_path, x=30, y=60, width=120, height=80)
    document = pypdfium2.PdfDocument(pdf_path)
    try:
        first = extract_image_assets(document)
        second = extract_image_assets(
            SimpleNamespace(pdfium_document=document, analyzed_page_count=1)
        )
    finally:
        document.close()

    assert first == second
    assert len(first) == 1
    asset = first[0]
    assert asset.page == 1
    assert asset.index == 1
    assert asset.pixel_size == (64, 48)
    assert asset.filters[-1] == "DCTDecode"
    assert asset.native_status == "available"
    assert asset.native_reason is None
    assert asset.format == "jpeg"
    assert asset.mime_type == "image/jpeg"
    assert asset.native_bytes == jpeg_bytes
    assert asset.sha256 == hashlib.sha256(jpeg_bytes).hexdigest()


def test_image_bbox_uses_top_left_pdf_coordinates(tmp_path: Path) -> None:
    jpeg_path = tmp_path / "positioned.jpg"
    Image.new("RGB", (20, 10), (30, 60, 90)).save(jpeg_path, format="JPEG")
    pdf_path = _image_pdf(tmp_path, jpeg_path, x=30, y=60, width=120, height=80)

    document = pypdfium2.PdfDocument(pdf_path)
    try:
        (asset,) = extract_image_assets(document)
    finally:
        document.close()

    assert asset.bbox.x0 == pytest.approx(30)
    assert asset.bbox.y0 == pytest.approx(260)
    assert asset.bbox.x1 == pytest.approx(150)
    assert asset.bbox.y1 == pytest.approx(340)
    assert asset.coverage == pytest.approx(0.08)


def test_lossless_pdf_stream_is_inventoried_without_reencoding(tmp_path: Path) -> None:
    png_path = tmp_path / "source.png"
    Image.new("RGB", (20, 10), (30, 60, 90)).save(png_path, format="PNG")
    pdf_path = _image_pdf(tmp_path, png_path, x=0, y=0, width=300, height=400)

    document = pypdfium2.PdfDocument(pdf_path)
    try:
        (asset,) = extract_image_assets(document)
    finally:
        document.close()

    assert asset.native_status == "unavailable"
    assert asset.native_reason == "no_standalone_native_encoding"
    assert asset.native_bytes is None
    assert asset.sha256 is None
    assert asset.format is None
    assert asset.mime_type is None

    document = pypdfium2.PdfDocument(pdf_path)
    try:
        decoded = decode_image_asset(document, asset, max_pixels=1_000)
    finally:
        document.close()

    assert decoded.source == "pdfium"
    assert decoded.image.mode == "RGB"
    assert decoded.image.size == (20, 10)


def test_vector_pdf_has_no_image_assets(tmp_path: Path) -> None:
    pdf_path = tmp_path / "vector.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(300, 400))
    pdf.setFillColorRGB(0.2, 0.4, 0.8)
    pdf.rect(30, 60, 120, 80, fill=1, stroke=0)
    pdf.drawString(30, 180, "Document vectoriel")
    pdf.showPage()
    pdf.save()

    document = pypdfium2.PdfDocument(pdf_path)
    try:
        assert extract_image_assets(document) == ()
    finally:
        document.close()


def test_image_inventory_is_bounded_and_reuses_repeated_native_bytes(tmp_path: Path) -> None:
    jpeg_path = tmp_path / "bounded.jpg"
    Image.new("RGB", (20, 10), (30, 60, 90)).save(jpeg_path, format="JPEG")
    pdf_path = tmp_path / "bounded.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(300, 400))
    pdf.drawImage(str(jpeg_path), 10, 10, width=100, height=50)
    pdf.drawImage(str(jpeg_path), 150, 200, width=100, height=50)
    pdf.showPage()
    pdf.save()

    document = pypdfium2.PdfDocument(pdf_path)
    try:
        limited = extract_image_assets(document, max_images=1)
        assets = extract_image_assets(document, max_images=2)
    finally:
        document.close()

    assert len(limited) == 1
    assert len(assets) == 2
    assert assets[0].index == 1
    assert assets[0].native_bytes is assets[1].native_bytes


def _image_pdf(
    tmp_path: Path,
    image_path: Path,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> Path:
    pdf_path = tmp_path / f"{image_path.stem}.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(300, 400))
    pdf.drawImage(str(image_path), x, y, width=width, height=height)
    pdf.showPage()
    pdf.save()
    return pdf_path
