"""Main-thread progress aggregation for concurrent analysis branches."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import TypeVar

ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class _ProgressEvent:
    branch: str
    value: float
    label: str


class ParallelProgress:
    """Combine branch progress while invoking the UI callback on the owner thread only."""

    def __init__(
        self,
        callback: Callable[[float, str], None] | None,
        *,
        start: float,
        end: float,
        weights: Mapping[str, float],
    ) -> None:
        if end < start:
            raise ValueError("end must not be lower than start")
        if not weights or any(weight <= 0 for weight in weights.values()):
            raise ValueError("branch weights must be positive")
        total_weight = sum(weights.values())
        self._callback = callback
        self._start = start
        self._width = end - start
        self._weights = {branch: weight / total_weight for branch, weight in weights.items()}
        self._values = {branch: 0.0 for branch in weights}
        self._events: SimpleQueue[_ProgressEvent] = SimpleQueue()
        self._last_global = start

    def worker_callback(self, branch: str) -> Callable[[float, str], None]:
        """Return a callback safe to invoke from a worker thread."""

        self._require_branch(branch)

        def enqueue(value: float, label: str) -> None:
            self._events.put(_ProgressEvent(branch, _bounded(value), label))

        return enqueue

    def update(self, branch: str, value: float, label: str) -> None:
        """Record owner-thread progress and flush queued worker events."""

        self._require_branch(branch)
        self._apply(_ProgressEvent(branch, _bounded(value), label))
        self.drain()

    def drain(self) -> None:
        """Publish every pending worker event from the owner thread."""

        while True:
            try:
                event = self._events.get_nowait()
            except Empty:
                return
            self._apply(event)

    def wait(self, future: Future[ResultT], *, poll_seconds: float = 0.05) -> ResultT:
        """Wait for one branch while keeping queued progress responsive."""

        while True:
            self.drain()
            try:
                result = future.result(timeout=poll_seconds)
            except TimeoutError:
                continue
            self.drain()
            return result

    def _apply(self, event: _ProgressEvent) -> None:
        current = self._values[event.branch]
        self._values[event.branch] = max(current, event.value)
        aggregate = self._start + self._width * sum(
            self._weights[branch] * value for branch, value in self._values.items()
        )
        aggregate = max(self._last_global, aggregate)
        self._last_global = aggregate
        if self._callback is not None:
            self._callback(aggregate, event.label)

    def _require_branch(self, branch: str) -> None:
        if branch not in self._weights:
            raise KeyError(f"Unknown progress branch: {branch}")


def _bounded(value: float) -> float:
    return min(1.0, max(0.0, value))
