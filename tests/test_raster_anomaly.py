import numpy as np
from PIL import Image, ImageDraw

from fraude_detector.detectors.raster_anomaly import analyze_jpeg_ela


def test_ela_returns_finite_page_sized_heatmap() -> None:
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 560, 400), outline="black", width=5)
    draw.text((120, 180), "DOCUMENT SYNTHETIQUE", fill="black")

    analysis = analyze_jpeg_ela(
        image=image,
        jpeg_quality=90,
        block_size=64,
        robust_z_threshold=3.5,
    )

    assert analysis.heatmap.shape == (480, 640)
    assert np.isfinite(analysis.heatmap).all()
    assert all(region.area_fraction > 0 for region in analysis.regions)
