from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from pypdf import PdfWriter
from reportlab.pdfgen import canvas


@pytest.fixture
def vector_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "vector.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.add_metadata(
        {
            "/Title": "Declaration synthetique",
            "/Producer": "Fraude Detector tests",
            "/CreationDate": "D:20260721120000+02'00'",
            "/ModDate": "D:20260721120000+02'00'",
        }
    )
    writer.write(path)
    return path


@pytest.fixture
def scan_pdf(tmp_path: Path) -> Path:
    image_path = tmp_path / "scan.jpg"
    image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 1120, 1520), outline="black", width=5)
    draw.text((130, 160), "DECLARATION ASSURANCE - DOCUMENT SYNTHETIQUE", fill="black")
    draw.text((130, 260), "Numero de dossier: TEST-2026-001", fill="black")
    draw.text((130, 340), "Montant declare: 1250 EUR", fill="black")
    image.save(image_path, format="JPEG", quality=82)

    pdf_path = tmp_path / "scan.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(600, 800))
    pdf.drawImage(str(image_path), 0, 0, width=600, height=800)
    pdf.showPage()
    pdf.save()
    return pdf_path
