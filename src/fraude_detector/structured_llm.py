"""Reliable JSON-only requests against a local vLLM server."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import requests


class StructuredLlmError(RuntimeError):
    """A structured LLM request did not produce one valid JSON object."""


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

    result = _response_object(response, operation)
    if result is not None:
        return result

    retry_payload = dict(payload)
    retry_payload["messages"] = _json_retry_messages(messages)
    retry_payload["max_tokens"] = min(max(max_tokens + 1, max_tokens * 2), 32_768)
    retry_response = _post(endpoint, retry_payload, timeout_seconds, operation)
    if getattr(retry_response, "status_code", 200) in {400, 422}:
        retry_payload["response_format"] = {"type": "json_object"}
        retry_response = _post(endpoint, retry_payload, timeout_seconds, operation)
    retry_result = _response_object(retry_response, operation)
    if retry_result is None:
        raise StructuredLlmError(
            f"{operation}: vLLM returned invalid or truncated JSON after a structured retry"
        )
    return retry_result


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


def _response_object(response: requests.Response, operation: str) -> Mapping[str, Any] | None:
    try:
        response.raise_for_status()
        envelope = response.json()
        choice = envelope["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError, requests.RequestException) as error:
        raise StructuredLlmError(f"{operation}: invalid vLLM response envelope: {error}") from error
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, Mapping) else None


def _json_retry_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    retried = [dict(message) for message in messages]
    instruction = (
        "A previous generation was invalid or truncated. Return exactly one complete JSON object "
        "matching the requested schema. Do not use Markdown fences, prose, comments, or trailing "
        "text. Keep free-text explanations concise, but do not omit extracted table rows."
    )
    if retried and retried[0].get("role") == "system":
        retried[0]["content"] = f"{retried[0]['content']}\n{instruction}"
    else:
        retried.insert(0, {"role": "system", "content": instruction})
    return retried
