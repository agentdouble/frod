from __future__ import annotations

import logging

from fraude_detector.pdf_logging import suppress_pypdf_recovery_messages


def test_pypdf_recovery_messages_are_scoped_and_errors_remain_visible(caplog) -> None:
    logger = logging.getLogger("pypdf")
    previous_level = logger.level

    with caplog.at_level(
        logging.WARNING, logger="pypdf"
    ), suppress_pypdf_recovery_messages():
        logger.warning("Multiple definitions in dictionary for key /Info")
        logger.error("PDF réellement illisible")

    assert logger.level == previous_level
    assert "Multiple definitions" not in caplog.text
    assert "PDF réellement illisible" in caplog.text
