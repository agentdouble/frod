from PIL import Image, ImageDraw

from fraude_detector.detectors.revision_diff import compare_revision_images


def test_revision_diff_localizes_changed_region() -> None:
    previous = Image.new("RGB", (600, 800), "white")
    current = previous.copy()
    draw = ImageDraw.Draw(current)
    draw.rectangle((180, 300, 360, 360), fill="black")

    findings, heatmap = compare_revision_images(
        previous,
        current,
        page_number=1,
        page_width=600,
        page_height=800,
    )

    assert findings
    assert heatmap is not None
    assert heatmap.size == current.size
    box = findings[0].bbox
    assert box is not None
    assert box.x0 <= 180 <= box.x1
    assert box.y0 <= 300 <= box.y1


def test_revision_diff_ignores_identical_images() -> None:
    image = Image.new("RGB", (200, 300), "white")
    findings, heatmap = compare_revision_images(image, image.copy(), 1, 200, 300)
    assert findings == []
    assert heatmap is None
