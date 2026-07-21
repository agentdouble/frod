from fraude_detector.models import Finding
from fraude_detector.scoring import assess_risk


def finding(category: str, points: float) -> Finding:
    return Finding(
        detector="test",
        code=f"TEST_{category}",
        category=category,
        title="Test",
        description="Synthetic test signal",
        risk_points=points,
        confidence=1.0,
    )


def test_one_family_never_reaches_high() -> None:
    assessment = assess_risk(tuple(finding("revision_visual", 60) for _ in range(5)))
    assert assessment.score == 60
    assert assessment.level == "review"


def test_repeated_regions_with_same_code_do_not_inflate_score() -> None:
    single = assess_risk((finding("revision_visual", 55),))
    fragmented = assess_risk(tuple(finding("revision_visual", 55) for _ in range(8)))

    assert fragmented.score == single.score


def test_independent_strong_families_can_reach_high() -> None:
    assessment = assess_risk(
        (
            finding("revision_visual", 60),
            finding("page_composition", 45),
            finding("raster_forensics", 30),
        )
    )
    assert assessment.score >= 70
    assert assessment.level == "high"


def test_no_signal_is_low_but_not_authenticity_claim() -> None:
    assessment = assess_risk(())
    assert assessment.level == "low"
    assert "authenticite" in assessment.explanation
