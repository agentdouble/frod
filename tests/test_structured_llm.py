from __future__ import annotations

from typing import Any

import pytest
import requests

from fraude_detector.structured_llm import StructuredLlmError, request_json_object


class _Response:
    def __init__(self, content: str, *, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": self.content}}]}


SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "test_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    },
}


def _request() -> dict[str, Any]:
    return dict(
        endpoint="http://vllm.test/v1/chat/completions",
        model="local-model",
        messages=[{"role": "system", "content": "Return JSON."}],
        response_format=SCHEMA,
        temperature=0.0,
        max_tokens=1000,
        timeout_seconds=30,
        operation="test operation",
    )


def test_invalid_json_is_retried_with_schema_and_more_output_tokens(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    responses = iter((_Response('{"value":'), _Response('{"value":"complete"}')))

    def fake_post(url: str, **kwargs: Any) -> _Response:
        calls.append({"url": url, **kwargs})
        return next(responses)

    monkeypatch.setattr(requests, "post", fake_post)
    result = request_json_object(**_request())

    assert result == {"value": "complete"}
    assert len(calls) == 2
    assert calls[0]["json"]["response_format"]["type"] == "json_schema"
    assert calls[1]["json"]["response_format"]["type"] == "json_schema"
    assert calls[1]["json"]["max_tokens"] == 2000
    assert "invalid or truncated" in calls[1]["json"]["messages"][0]["content"]


def test_no_attempt_ever_falls_back_to_free_text(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    responses = iter(
        (
            _Response("", status_code=400),
            _Response("not json"),
            _Response("still not json"),
        )
    )

    def fake_post(url: str, **kwargs: Any) -> _Response:
        calls.append(kwargs["json"])
        return next(responses)

    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(StructuredLlmError, match="invalid or truncated JSON"):
        request_json_object(**_request())

    assert [call["response_format"]["type"] for call in calls] == [
        "json_schema",
        "json_object",
        "json_object",
    ]
