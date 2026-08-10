from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import pkcs12
from cryptography.x509.oid import NameOID
from pyhanko.pdf_utils import generic
from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
from pyhanko.sign import signers
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from fraude_detector.config import AnalysisConfig
from fraude_detector.laboratory import analyze_pdf_laboratory
from fraude_detector.laboratory.two_d_doc import _verify_payload


def test_laboratory_compares_every_retained_revision(tmp_path: Path) -> None:
    source_path = Path("tests/fixtures/assurance-fraude.pdf")
    three_revision_path = tmp_path / "three-revisions.pdf"
    with source_path.open("rb") as source:
        writer = IncrementalPdfFileWriter(source)
        stream = generic.StreamObject(stream_data=b"q 0 0 1 rg 80 80 90 30 re f Q")
        writer.add_stream_to_page(0, writer.add_object(stream))
        with three_revision_path.open("wb") as output:
            writer.write(output)

    report = analyze_pdf_laboratory(
        three_revision_path,
        tmp_path,
        config=AnalysisConfig(max_pages=5),
    )

    checks = {check.code: check for check in report.checks}
    assert len(checks) == 6
    assert checks["all_revisions"].state == "detected"
    assert "3 revisions" in checks["all_revisions"].summary
    assert len(checks["all_revisions"].observations) == 2
    assert checks["pades"].state == "not_applicable"
    assert checks["two_d_doc"].state == "not_applicable"


def test_named_but_malformed_facturx_attachment_is_reported(tmp_path: Path) -> None:
    pdf_path = tmp_path / "malformed-facturx.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.add_attachment("factur-x.xml", b"<CrossIndustryInvoice>")
    writer.write(pdf_path)

    report = analyze_pdf_laboratory(
        pdf_path,
        tmp_path / "output",
        config=AnalysisConfig(max_pages=1),
    )

    check = next(item for item in report.checks if item.code == "facturx")
    assert check.state == "attention"
    assert [item.code for item in check.observations] == ["FACTURX_XML_MALFORMED"]
    assert check.observations[0].strength == "moderate"


def test_self_signed_pdf_signature_is_intact_but_not_locally_trusted(
    tmp_path: Path,
) -> None:
    pdf_path = _create_self_signed_pdf(tmp_path)

    report = analyze_pdf_laboratory(
        pdf_path,
        tmp_path / "output",
        config=AnalysisConfig(max_pages=1),
    )

    checks = {check.code: check for check in report.checks}
    signature = checks["pades"].observations[0]
    assert signature.code == "PDF_SIGNATURE_INTACT_UNTRUSTED"
    assert signature.evidence["cryptographically_intact"] is True
    assert signature.evidence["cryptographic_signature_valid"] is True
    assert signature.evidence["certificate_trusted_locally"] is False
    assert checks["post_signature"].state == "clear"


def test_visible_change_after_signature_is_reported_as_strong(
    tmp_path: Path,
) -> None:
    signed_path = _create_self_signed_pdf(tmp_path)
    modified_path = tmp_path / "signed-then-modified.pdf"
    with signed_path.open("rb") as source:
        writer = IncrementalPdfFileWriter(source)
        stream = generic.StreamObject(stream_data=b"q 1 0 0 rg 100 100 160 40 re f Q")
        writer.add_stream_to_page(0, writer.add_object(stream))
        with modified_path.open("wb") as output:
            writer.write(output)

    report = analyze_pdf_laboratory(
        modified_path,
        tmp_path / "modified-output",
        config=AnalysisConfig(max_pages=1),
    )

    check = next(item for item in report.checks if item.code == "post_signature")
    assert check.state == "attention"
    assert check.observations[0].code == "PDF_SUSPICIOUS_POST_SIGNATURE_CHANGE"
    assert check.observations[0].strength == "strong"


def test_rare_numeric_font_and_unexplained_hidden_text_are_weak_signals(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "font-anomalies.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(595, 842))
    for index in range(10):
        pdf.setFont("Helvetica", 11)
        pdf.drawString(60, 780 - index * 28, f"Ligne de facture {index}")
    pdf.setFont("Courier", 11)
    pdf.drawString(360, 500, "9999,99")
    hidden = pdf.beginText(60, 450)
    hidden.setTextRenderMode(3)
    hidden.textLine("reference cachee 12345")
    pdf.drawText(hidden)
    pdf.save()

    report = analyze_pdf_laboratory(
        pdf_path,
        tmp_path / "font-output",
        config=AnalysisConfig(max_pages=1),
    )

    check = next(item for item in report.checks if item.code == "fonts_hidden_objects")
    codes = {item.code for item in check.observations}
    assert "PDF_RARE_FONT_NUMERIC_FRAGMENT" in codes
    assert "PDF_UNEXPLAINED_INVISIBLE_TEXT" in codes
    rare_font = next(
        item for item in check.observations if item.code == "PDF_RARE_FONT_NUMERIC_FRAGMENT"
    )
    assert rare_font.state == "detected"
    assert rare_font.strength == "informational"
    hidden_text = next(
        item for item in check.observations if item.code == "PDF_UNEXPLAINED_INVISIBLE_TEXT"
    )
    assert hidden_text.strength == "weak"


def test_malformed_two_d_doc_payload_is_a_moderate_signal() -> None:
    observation = _verify_payload(b"DC03not-a-complete-2d-doc", page=2)

    assert observation.code == "TWO_D_DOC_MALFORMED"
    assert observation.state == "attention"
    assert observation.strength == "moderate"
    assert observation.page == 2


def _create_self_signed_pdf(tmp_path: Path) -> Path:
    base_path = tmp_path / "unsigned.pdf"
    pdf = canvas.Canvas(str(base_path), pagesize=(595, 842))
    pdf.drawString(60, 780, "Document de test signe")
    pdf.save()

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Frod test signer")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=True,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .sign(private_key, hashes.SHA256())
    )
    archive = pkcs12.serialize_key_and_certificates(
        name=b"frod-test",
        key=private_key,
        cert=certificate,
        cas=None,
        encryption_algorithm=serialization.NoEncryption(),
    )
    signer = signers.SimpleSigner.load_pkcs12_data(
        archive,
        other_certs=(),
        passphrase=None,
    )
    assert signer is not None

    signed_path = tmp_path / "signed.pdf"
    with base_path.open("rb") as source:
        incremental_writer = IncrementalPdfFileWriter(source)
        output = io.BytesIO()
        signers.sign_pdf(
            incremental_writer,
            signers.PdfSignatureMetadata(field_name="Signature1"),
            signer=signer,
            output=output,
        )
    signed_path.write_bytes(output.getvalue())
    return signed_path
