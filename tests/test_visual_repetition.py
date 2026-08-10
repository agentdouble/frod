from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from fraude_detector.laboratory.visual_repetition import analyze_repeated_visual_regions


def test_three_identical_lower_page_marks_are_reported_without_scoring(tmp_path: Path) -> None:
    page_paths = tuple(_page_with_mark(tmp_path, page) for page in range(1, 4))
    payload = [
        [
            {
                "label": "signature",
                "content": "",
                "bbox_2d": [600, 700, 900, 850],
            }
        ]
        for _ in range(3)
    ]

    report = analyze_repeated_visual_regions(payload, page_paths, tmp_path / "laboratory")

    assert report.state == "detected"
    observation = report.observations[0]
    assert observation.code == "OCR_REPEATED_VISUAL_REGION"
    assert observation.strength == "weak"
    assert observation.evidence["pages"] == (1, 2, 3)
    assert observation.evidence["scored_by_frod"] is False
    assert (tmp_path / "laboratory" / observation.artifacts[0]).is_file()


def test_repeated_header_logos_are_outside_the_comparison_scope(tmp_path: Path) -> None:
    page_paths = tuple(_page_with_mark(tmp_path, page) for page in range(1, 4))
    payload = [[{"label": "image", "content": "", "bbox_2d": [50, 50, 250, 150]}] for _ in range(3)]

    report = analyze_repeated_visual_regions(payload, page_paths, tmp_path / "laboratory")

    assert report.state == "not_applicable"


def test_logo_and_signature_on_the_same_pages_are_kept_as_distinct_groups(
    tmp_path: Path,
) -> None:
    page_paths = tuple(_page_with_two_marks(tmp_path, page) for page in range(1, 4))
    payload = [
        [
            {"label": "image", "content": "", "bbox_2d": [500, 600, 620, 700]},
            {"label": "image", "content": "", "bbox_2d": [650, 760, 900, 880]},
        ]
        for _ in range(3)
    ]

    report = analyze_repeated_visual_regions(payload, page_paths, tmp_path / "laboratory")

    assert report.state == "detected"
    assert len(report.observations) == 2
    centers = {
        round(sum(box[0] + box[2] for box in item.evidence["normalized_boxes"]) / 6)
        for item in report.observations
    }
    assert centers == {560, 775}


def _page_with_mark(root: Path, page: int) -> Path:
    image = Image.new("RGB", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.line((650, 790, 700, 740, 760, 810, 840, 755), fill="black", width=8)
    draw.arc((690, 730, 850, 825), 10, 175, fill="black", width=5)
    path = root / f"page-{page}.png"
    image.save(path)
    return path


def _page_with_two_marks(root: Path, page: int) -> Path:
    image = Image.new("RGB", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((520, 620, 600, 680), outline="black", width=8)
    draw.line((680, 840, 730, 780, 790, 850, 860, 795), fill="black", width=8)
    draw.arc((700, 770, 875, 860), 10, 175, fill="black", width=5)
    path = root / f"two-marks-{page}.png"
    image.save(path)
    return path
