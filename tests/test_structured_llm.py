from __future__ import annotations

import json
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


class _StreamingResponse:
    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, *, decode_unicode: bool) -> list[str]:
        assert decode_unicode is True
        return [f"data: {json.dumps(chunk)}" for chunk in self.chunks] + ["data: [DONE]"]

    def close(self) -> None:
        return None


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
    assert all(
        call["json"]["chat_template_kwargs"] == {"enable_thinking": False}
        for call in calls
    )
    assert all("reasoning_effort" not in call["json"] for call in calls)
    assert all(call["json"]["stream"] is True for call in calls)
    assert all(call["stream"] is True for call in calls)


def test_streaming_prints_reasoning_and_final_content_with_ansi(
    monkeypatch: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    response = _StreamingResponse(
        [
            {"choices": [{"delta": {"reasoning_content": "Vérification... "}}]},
            {"choices": [{"delta": {"content": '{"value":'}}]},
            {
                "choices": [
                    {"delta": {"content": '"complete"}'}, "finish_reason": "stop"}
                ]
            },
        ]
    )
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)

    assert request_json_object(**_request()) == {"value": "complete"}

    terminal = capsys.readouterr().out
    assert "\033[1;36m━━ TEST OPERATION ━━" in terminal
    assert "\033[3;90mVérification... " in terminal
    assert '{"value":' in terminal
    assert '"complete"}' in terminal
    assert "\033[1;32m✓ test operation terminé en" in terminal
    assert "20 caractères" in terminal


def test_reasoning_effort_is_forwarded_to_template_on_every_attempt(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []
    responses = iter((_Response('{"value":'), _Response('{"value":"complete"}')))

    def fake_post(url: str, **kwargs: Any) -> _Response:
        calls.append({"url": url, **kwargs})
        return next(responses)

    monkeypatch.setattr(requests, "post", fake_post)
    request = _request()
    request["reasoning_effort"] = "low"

    assert request_json_object(**request) == {"value": "complete"}
    assert all("reasoning_effort" not in call["json"] for call in calls)
    assert all(
        call["json"]["chat_template_kwargs"]
        == {"enable_thinking": True, "reasoning_effort": "low"}
        for call in calls
    )


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
