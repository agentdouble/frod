"""Scoped suppression of noisy pypdf recovery messages."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock

_PYPDF_LOG_LOCK = RLock()


@contextmanager
def suppress_pypdf_recovery_messages() -> Iterator[None]:
    """Hide tolerant-parser warnings while preserving pypdf errors."""

    with _PYPDF_LOG_LOCK:
        logger = logging.getLogger("pypdf")
        previous_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            yield
        finally:
            logger.setLevel(previous_level)
