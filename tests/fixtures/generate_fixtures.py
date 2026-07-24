"""Generate the two synthetic insurance PDFs committed as integration fixtures."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 1697
AMOUNT_REGION = (96, 600, 900, 700)


def generate_fixtures(output_dir: Path) -> tuple[Path, Path, Path]:
    """Create intact, altered and legitimate-update insurance fixtures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    temporary_root = repo_root / "tmp" / "pdfs"
    temporary_root.mkdir(parents=True, exist_ok=True)

    clean_path = output_dir / "assurance-sans-fraude.pdf"
    fraud_path = output_dir / "assurance-fraude.pdf"
    legitimate_path = output_dir / "assurance-ajout-legitime.pdf"

    with tempfile.TemporaryDirectory(prefix="fixture-generation-", dir=temporary_root) as directory:
        work_dir = Path(directory)
        scan_path = work_dir / "assurance-scan.jpg"
        initial_pdf = work_dir / "assurance-initial.pdf"
        overlay_pdf = work_dir / "montant-modifie-overlay.pdf"
        stamp_path = work_dir / "tampon-ajoute.png"
        legitimate_overlay_pdf = work_dir / "reception-overlay.pdf"
        legitimate_stamp_path = work_dir / "tampon-reception.png"

        _create_scan(scan_path)
        _embed_scan(scan_path, initial_pdf)
        _write_canonical_clean_pdf(initial_pdf, clean_path)
        _create_stamp(stamp_path)
        _create_modified_amount_overlay(overlay_pdf, stamp_path)
        _write_incremental_fraud_pdf(clean_path, overlay_pdf, fraud_path)
        _create_legitimate_stamp(legitimate_stamp_path)
        _create_legitimate_overlay(legitimate_overlay_pdf, legitimate_stamp_path)
        _write_incremental_legitimate_pdf(
            clean_path,
            legitimate_overlay_pdf,
            legitimate_path,
        )

    return clean_path, fraud_path, legitimate_path


def _create_scan(path: Path) -> None:
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "#F8FAFC")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=48)
    subtitle_font = ImageFont.load_default(size=28)
    body_font = ImageFont.load_default(size=34)
    amount_font = ImageFont.load_default(size=42)
    small_font = ImageFont.load_default(size=24)

    draw.rounded_rectangle(
        (55, 55, 1145, 1642), radius=20, fill="white", outline="#0F172A", width=5
    )
    draw.rectangle((55, 55, 1145, 245), fill="#123B66")
    draw.text((105, 105), "DECLARATION DE SINISTRE", font=title_font, fill="white")
    draw.text(
        (108, 175),
        "Document synthetique de validation - aucune donnee reelle",
        font=subtitle_font,
        fill="#DCEAF7",
    )

    draw.text((105, 330), "Assure", font=subtitle_font, fill="#475569")
    draw.text((105, 380), "Camille Exemple", font=body_font, fill="#0F172A")
    draw.text((650, 330), "Dossier", font=subtitle_font, fill="#475569")
    draw.text((650, 380), "TEST-2026-001", font=body_font, fill="#0F172A")

    draw.rounded_rectangle(AMOUNT_REGION, radius=14, fill="#EEF2F7", outline="#CBD5E1", width=3)
    draw.text(
        (125, 625),
        "Montant des dommages : 1 250 EUR",
        font=amount_font,
        fill="#111827",
    )

    rows = (
        ("Date du sinistre", "18/07/2026"),
        ("Type de dommage", "Degat des eaux"),
        ("Reference contrat", "HAB-TEST-042"),
    )
    start_y = 820
    for index, (label, value) in enumerate(rows):
        y = start_y + index * 150
        draw.text((110, y), label, font=subtitle_font, fill="#64748B")
        draw.text((110, y + 48), value, font=body_font, fill="#0F172A")
        draw.line((105, y + 110, 1095, y + 110), fill="#E2E8F0", width=2)

    draw.rounded_rectangle((95, 1370, 1105, 1530), radius=12, outline="#94A3B8", width=2)
    draw.text((120, 1405), "Observation", font=subtitle_font, fill="#475569")
    draw.text(
        (120, 1460),
        "Fixture creee uniquement pour tester la pipeline Frod.",
        font=small_font,
        fill="#334155",
    )
    draw.text((830, 1580), "PAGE 1 / 1", font=small_font, fill="#64748B")

    image.save(path, format="JPEG", quality=84, optimize=False, progressive=False)


def _embed_scan(scan_path: Path, output_path: Path) -> None:
    page_width, page_height = A4
    pdf = canvas.Canvas(
        str(output_path),
        pagesize=A4,
        invariant=1,
        pageCompression=1,
    )
    pdf.setTitle("Declaration assurance synthetique")
    pdf.setAuthor("Frod test fixtures")
    pdf.setCreator("Frod fixture generator")
    pdf.drawImage(
        str(scan_path),
        0,
        0,
        width=page_width,
        height=page_height,
        preserveAspectRatio=False,
    )
    pdf.showPage()
    pdf.save()


def _write_canonical_clean_pdf(source_path: Path, output_path: Path) -> None:
    source = PdfReader(source_path)
    writer = PdfWriter()
    for page in source.pages:
        writer.add_page(page)
    writer.add_metadata(
        {
            "/Title": "Declaration assurance synthetique",
            "/Author": "Frod test fixtures",
            "/Creator": "Frod fixture generator",
            "/Producer": "Frod fixture generator",
            "/CreationDate": "D:20260721120000+02'00'",
            "/ModDate": "D:20260721120000+02'00'",
        }
    )
    writer.write(output_path)


def _create_stamp(output_path: Path) -> None:
    stamp = Image.new("RGBA", (360, 180), (255, 255, 255, 0))
    draw = ImageDraw.Draw(stamp)
    font = ImageFont.load_default(size=48)
    draw.rounded_rectangle(
        (8, 8, 352, 172),
        radius=28,
        fill=(255, 255, 255, 235),
        outline="#B91C1C",
        width=12,
    )
    draw.text((74, 60), "VALIDE", font=font, fill="#B91C1C")
    stamp.save(output_path, format="PNG")


def _create_legitimate_stamp(output_path: Path) -> None:
    stamp = Image.new("RGBA", (420, 170), (255, 255, 255, 0))
    draw = ImageDraw.Draw(stamp)
    title_font = ImageFont.load_default(size=42)
    detail_font = ImageFont.load_default(size=24)
    draw.rounded_rectangle(
        (8, 8, 412, 162),
        radius=22,
        fill=(255, 255, 255, 220),
        outline="#0369A1",
        width=10,
    )
    draw.text((102, 38), "RECU", font=title_font, fill="#0369A1")
    draw.text((67, 96), "23/07/2026 - Service sinistres", font=detail_font, fill="#075985")
    stamp.save(output_path, format="PNG")


def _create_modified_amount_overlay(output_path: Path, stamp_path: Path) -> None:
    page_width, page_height = A4
    x0, y0, x1, y1 = AMOUNT_REGION
    region_x = x0 / IMAGE_WIDTH * page_width
    region_y = (IMAGE_HEIGHT - y1) / IMAGE_HEIGHT * page_height
    region_width = (x1 - x0) / IMAGE_WIDTH * page_width
    region_height = (y1 - y0) / IMAGE_HEIGHT * page_height

    pdf = canvas.Canvas(
        str(output_path),
        pagesize=A4,
        invariant=1,
        pageCompression=1,
    )
    pdf.setFillColor(HexColor("#EEF2F7"))
    pdf.roundRect(
        region_x,
        region_y,
        region_width,
        region_height,
        radius=7,
        stroke=0,
        fill=1,
    )
    pdf.setFillColor(HexColor("#111827"))
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(region_x + 14, region_y + 30, "Montant des dommages : 9 500 EUR")
    pdf.drawImage(
        str(stamp_path),
        465,
        region_y + 3,
        width=90,
        height=45,
        mask="auto",
    )
    pdf.showPage()
    pdf.save()


def _create_legitimate_overlay(output_path: Path, stamp_path: Path) -> None:
    pdf = canvas.Canvas(
        str(output_path),
        pagesize=A4,
        invariant=1,
        pageCompression=1,
    )
    pdf.drawImage(
        str(stamp_path),
        380,
        55,
        width=160,
        height=65,
        mask="auto",
    )
    pdf.showPage()
    pdf.save()


def _write_incremental_fraud_pdf(
    clean_path: Path,
    overlay_path: Path,
    output_path: Path,
) -> None:
    writer = PdfWriter(clean_path, incremental=True)
    overlay = PdfReader(overlay_path)
    writer.pages[0].merge_page(overlay.pages[0])
    writer.add_metadata(
        {
            "/ModDate": "D:20260721123000+02'00'",
            "/FixtureMutation": "Montant remplace de 1 250 EUR par 9 500 EUR",
        }
    )
    writer.write(output_path)


def _write_incremental_legitimate_pdf(
    clean_path: Path,
    overlay_path: Path,
    output_path: Path,
) -> None:
    writer = PdfWriter(clean_path, incremental=True)
    overlay = PdfReader(overlay_path)
    writer.pages[0].merge_page(overlay.pages[0])
    writer.add_metadata(
        {
            "/ModDate": "D:20260723100000+02'00'",
            "/FixtureMutation": "Tampon de reception ajoute sans masquer le contenu",
        }
    )
    writer.write(output_path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    paths = generate_fixtures(args.output_dir.resolve())
    for path in paths:
        print(f"{path}: {_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
