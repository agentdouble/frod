from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fraude_detector import trufor
from fraude_detector.laboratory import images
from fraude_detector.models import LaboratoryReport


def test_trufor_result_is_validated_and_rendered(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "image.png"
    Image.new("RGB", (80, 60), "white").save(source)
    weights = tmp_path / "trufor.pth.tar"
    weights.write_bytes(b"test")

    monkeypatch.setattr(trufor, "_validate_checkpoint", lambda path: None)

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        anomaly = np.zeros((60, 80), dtype=np.float32)
        anomaly[20:40, 30:50] = 0.9
        confidence = np.full((60, 80), 0.8, dtype=np.float32)
        np.savez(
            output,
            map=anomaly,
            conf=confidence,
            score=np.asarray([0.82], dtype=np.float32),
            original_size=np.asarray([80, 60], dtype=np.int32),
            analyzed_size=np.asarray([80, 60], dtype=np.int32),
        )
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(trufor.subprocess, "run", fake_run)
    analysis = trufor.analyze_trufor_image(
        source,
        tmp_path / "result",
        weights_path=weights,
    )

    assert analysis.score == np.float32(0.82)
    assert analysis.reliable_suspect_ratio == 400 / 4800
    assert not analysis.resized
    assert analysis.artifacts == (
        "trufor/trufor-localization-map.png",
        "trufor/trufor-reliable-map.png",
        "trufor/trufor-confidence.png",
    )
    for relative in analysis.artifacts:
        assert (tmp_path / "result" / "laboratory" / relative).is_file()


def test_image_laboratory_keeps_high_trufor_score_outside_frod(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analysis = trufor.TruForAnalysis(
        score=0.91,
        anomaly_map=np.zeros((10, 10), dtype=np.float32),
        confidence_map=np.ones((10, 10), dtype=np.float32),
        reliable_anomaly_map=np.zeros((10, 10), dtype=np.float32),
        original_size=(10, 10),
        analyzed_size=(10, 10),
        resized=False,
        reliable_suspect_ratio=0.03,
        artifacts=("trufor/trufor-localization-map.png",),
    )
    monkeypatch.setattr(images, "analyze_trufor_image", lambda *args, **kwargs: analysis)

    report = images.analyze_image_laboratory(
        tmp_path / "image.png",
        tmp_path / "result",
    )

    assert isinstance(report, LaboratoryReport)
    check = report.checks[0]
    assert check.state == "attention"
    assert "91%" in check.summary
    assert check.observations[0].evidence["scored_by_frod"] is False


def test_image_laboratory_reports_missing_trufor_without_crashing(tmp_path: Path) -> None:
    report = images.analyze_image_laboratory(
        tmp_path / "image.png",
        tmp_path / "result",
        trufor_weights=tmp_path / "missing.pth.tar",
    )

    assert report.checks[0].state == "error"
    assert report.checks[0].observations[0].code == "TRUFOR_UNAVAILABLE"


def test_worker_resize_respects_pixel_budget() -> None:
    from fraude_detector.trufor_worker import _resize_to_budget

    image = Image.new("RGB", (2000, 1000), "white")
    resized = _resize_to_budget(image, max_pixels=500_000)

    assert resized.size == (1000, 500)


def test_spectral_map_uses_blue_for_low_and_red_for_high() -> None:
    values = np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32)
    rendered = np.asarray(trufor._render_spectral_map(values))

    assert tuple(rendered[0, 0]) == (5, 48, 97)
    assert tuple(rendered[0, 1]) == (247, 247, 247)
    assert tuple(rendered[0, 2]) == (103, 0, 31)
