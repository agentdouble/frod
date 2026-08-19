from __future__ import annotations

from typing import Any

import pytest
import requests

from fraude_detector.structured_llm import StructuredLlmError, request_json_object


class _Response:
    def __init__(
        self,
        content: str,
        *,
        status_code: int = 200,
        finish_reason: str | None = None,
    ) -> None:
        self.content = content
        self.status_code = status_code
        self.finish_reason = finish_reason

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {"content": self.content},
                    "finish_reason": self.finish_reason,
                }
            ]
        }


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
    with pytest.raises(StructuredLlmError, match="invalid JSON"):
        request_json_object(**_request())

    assert [call["response_format"]["type"] for call in calls] == [
        "json_schema",
        "json_object",
        "json_object",
    ]


def test_markdown_or_prose_wrapper_does_not_discard_a_valid_json_object(
    monkeypatch: Any,
) -> None:
    calls = 0

    def fake_post(url: str, **kwargs: Any) -> _Response:
        nonlocal calls
        del url, kwargs
        calls += 1
        return _Response('Résultat:\n```json\n{"value":"wrapped"}\n```')

    monkeypatch.setattr(requests, "post", fake_post)

    assert request_json_object(**_request()) == {"value": "wrapped"}
    assert calls == 1


def test_truncation_is_reported_separately_with_non_sensitive_diagnostics(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _Response('{"value":', finish_reason="length"),
    )

    with pytest.raises(StructuredLlmError, match="finish_reason=length") as caught:
        request_json_object(**_request())

    assert caught.value.truncated is True
    assert "output_chars=9" in str(caught.value)


def test_a_nested_object_from_a_truncated_outer_object_is_never_accepted(
    monkeypatch: Any,
) -> None:
    truncated = '{"items":[{"value":"partial"}]'
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _Response(truncated, finish_reason="length"),
    )

    with pytest.raises(StructuredLlmError) as caught:
        request_json_object(**_request())

    assert caught.value.truncated is True


def test_json_missing_required_top_level_fields_is_retried(monkeypatch: Any) -> None:
    calls = 0

    def fake_post(url: str, **kwargs: Any) -> _Response:
        nonlocal calls
        del url, kwargs
        calls += 1
        if calls == 1:
            return _Response('{"unexpected":"value"}')
        return _Response('{"value":"complete"}')

    monkeypatch.setattr(requests, "post", fake_post)

    assert request_json_object(**_request()) == {"value": "complete"}
    assert calls == 2
