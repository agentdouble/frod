"""Reliable JSON-only requests against a local vLLM server."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import requests


class StructuredLlmError(RuntimeError):
    """A structured LLM request did not produce one valid JSON object."""

    def __init__(self, message: str, *, truncated: bool = False) -> None:
        super().__init__(message)
        self.truncated = truncated


@dataclass(frozen=True, slots=True)
class _DecodedResponse:
    result: Mapping[str, Any] | None
    finish_reason: str | None
    content_length: int
    truncated: bool


def request_json_object(
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    operation: str,
) -> Mapping[str, Any]:
    """Return one JSON object, retaining constrained output on every attempt."""

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format,
    }
    response = _post(endpoint, payload, timeout_seconds, operation)
    if getattr(response, "status_code", 200) in {400, 422}:
        # Older vLLM versions may reject json_schema while still supporting JSON mode.
        payload["response_format"] = {"type": "json_object"}
        response = _post(endpoint, payload, timeout_seconds, operation)

    decoded = _response_object(response, operation)
    if decoded.result is not None and _has_required_fields(decoded.result, response_format):
        return decoded.result

    retry_payload = dict(payload)
    retry_payload["messages"] = _json_retry_messages(messages)
    retry_payload["max_tokens"] = min(max(max_tokens + 1, max_tokens * 2), 32_768)
    retry_response = _post(endpoint, retry_payload, timeout_seconds, operation)
    if getattr(retry_response, "status_code", 200) in {400, 422}:
        retry_payload["response_format"] = {"type": "json_object"}
        retry_response = _post(endpoint, retry_payload, timeout_seconds, operation)
    retry_decoded = _response_object(retry_response, operation)
    if retry_decoded.result is None or not _has_required_fields(
        retry_decoded.result, response_format
    ):
        truncated = decoded.truncated or retry_decoded.truncated
        reason = retry_decoded.finish_reason or decoded.finish_reason or "unknown"
        state = "truncated" if truncated else "invalid"
        raise StructuredLlmError(
            f"{operation}: vLLM returned {state} JSON after a structured retry "
            f"(finish_reason={reason}, output_chars={retry_decoded.content_length})",
            truncated=truncated,
        )
    return retry_decoded.result


def _post(
    endpoint: str,
    payload: dict[str, Any],
    timeout_seconds: int,
    operation: str,
) -> requests.Response:
    try:
        return requests.post(endpoint, json=dict(payload), timeout=timeout_seconds)
    except requests.RequestException as error:
        raise StructuredLlmError(f"{operation}: local vLLM request failed: {error}") from error


def _response_object(response: requests.Response, operation: str) -> _DecodedResponse:
    try:
        response.raise_for_status()
        envelope = response.json()
        choice = envelope["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, ValueError, requests.RequestException) as error:
        raise StructuredLlmError(f"{operation}: invalid vLLM response envelope: {error}") from error
    normalized_reason = str(finish_reason) if finish_reason is not None else None
    if not isinstance(content, str) or not content.strip():
        return _DecodedResponse(
            result=None,
            finish_reason=normalized_reason,
            content_length=0,
            truncated=normalized_reason in {"length", "max_tokens"},
        )
    result = _decode_json_object(content)
    stripped = content.rstrip()
    looks_cut = stripped.lstrip().startswith("{") and not stripped.endswith("}")
    return _DecodedResponse(
        result=result,
        finish_reason=normalized_reason,
        content_length=len(content),
        truncated=normalized_reason in {"length", "max_tokens"} or (result is None and looks_cut),
    )


def _decode_json_object(content: str) -> Mapping[str, Any] | None:
    """Decode a JSON object, tolerating an accidental prose or Markdown wrapper."""

    stripped = content.strip()
    try:
        direct = json.loads(stripped)
    except json.JSONDecodeError:
        direct = None
    if isinstance(direct, Mapping):
        return direct

    object_start = stripped.find("{")
    if object_start < 0:
        return None
    try:
        candidate, _ = json.JSONDecoder().raw_decode(stripped, object_start)
    except json.JSONDecodeError:
        return None
    return candidate if isinstance(candidate, Mapping) else None


def _has_required_fields(
    result: Mapping[str, Any],
    response_format: Mapping[str, Any],
) -> bool:
    schema_container = response_format.get("json_schema")
    if not isinstance(schema_container, Mapping):
        return True
    schema = schema_container.get("schema")
    if not isinstance(schema, Mapping):
        return True
    required = schema.get("required")
    if not isinstance(required, list):
        return True
    return all(isinstance(key, str) and key in result for key in required)


def _json_retry_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    retried = [dict(message) for message in messages]
    instruction = (
        "A previous generation was invalid or truncated. Return exactly one complete JSON object "
        "matching the requested schema. Do not use Markdown fences, prose, comments, or trailing "
        "text. Return only fields required by the schema and keep every free-text value concise."
    )
    if retried and retried[0].get("role") == "system":
        retried[0]["content"] = f"{retried[0]['content']}\n{instruction}"
    else:
        retried.insert(0, {"role": "system", "content": instruction})
    return retried
