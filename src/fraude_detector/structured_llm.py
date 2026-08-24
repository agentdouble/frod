"""Observable structured and text requests against a local vLLM server."""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import requests

_ANSI_RESET = "\033[0m"
_ANSI_HEADER = "\033[1;36m"
_ANSI_THINKING = "\033[3;90m"
_ANSI_SUCCESS = "\033[1;32m"
_ANSI_WARNING = "\033[1;33m"

_OPERATION_LABELS = {
    "document classification": "CLASSIFICATION",
    "document extraction": "EXTRACTION",
    "extraction verification": "VÉRIFICATION",
    "analysis synthesis": "SYNTHÈSE",
}


class StructuredLlmError(RuntimeError):
    """A structured LLM request did not produce one valid JSON object."""

    def __init__(self, message: str, *, truncated: bool = False) -> None:
        super().__init__(message)
        self.truncated = truncated


@dataclass(frozen=True, slots=True)
class _CompletionMessage:
    content: str
    reasoning_content: str
    finish_reason: str | None


def request_text_completion(
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    operation: str,
) -> str:
    """Return the final plain-text answer from one local vLLM request."""

    response = _post(
        endpoint,
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout_seconds,
        operation,
    )
    try:
        response.raise_for_status()
        completion = _completion_message(response, operation)
    except requests.RequestException as error:
        raise StructuredLlmError(f"{operation}: invalid vLLM response envelope: {error}") from error

    text = completion.content.strip()
    # This also supports a MiniMax server started without its reasoning parser.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if not text:
        truncated = str(completion.finish_reason) in {"length", "max_tokens"}
        detail = "final answer missing after reasoning"
        if truncated:
            detail += " (token limit reached)"
        raise StructuredLlmError(f"{operation}: {detail}", truncated=truncated)
    return text


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
    reasoning_effort: str | None = None,
) -> Mapping[str, Any]:
    """Return one JSON object, retaining constrained output on every attempt."""

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format,
    }
    if reasoning_effort is not None:
        payload["chat_template_kwargs"] = {
            "enable_thinking": True,
            "reasoning_effort": reasoning_effort,
        }
    response = _post(endpoint, payload, timeout_seconds, operation)
    if getattr(response, "status_code", 200) in {400, 422}:
        _terminal_rejected(response, "format JSON Schema refusé, nouvel essai en mode JSON")
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
        _terminal_rejected(
            retry_response,
            "format JSON Schema refusé pendant la reprise, nouvel essai en mode JSON",
        )
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
    body = dict(payload)
    template_kwargs = dict(body.get("chat_template_kwargs", {}))
    template_kwargs.pop("thinking_budget", None)
    template_kwargs.setdefault("enable_thinking", False)
    body["chat_template_kwargs"] = template_kwargs
    body["stream"] = True
    _terminal_header(operation)
    started_at = time.monotonic()
    try:
        response = requests.post(
            endpoint,
            json=body,
            timeout=timeout_seconds,
            stream=True,
        )
        response._frod_started_at = started_at  # type: ignore[attr-defined]
        return response
    except requests.RequestException as error:
        raise StructuredLlmError(f"{operation}: local vLLM request failed: {error}") from error


def _completion_message(response: requests.Response, operation: str) -> _CompletionMessage:
    headers = getattr(response, "headers", {})
    content_type = headers.get("content-type", "") if isinstance(headers, Mapping) else ""
    if "text/event-stream" in str(content_type).casefold():
        completion = _streamed_completion(response, operation)
    else:
        completion = _buffered_completion(response, operation)
    _terminal_success(response, operation, len(completion.content))
    return completion


def _streamed_completion(response: requests.Response, operation: str) -> _CompletionMessage:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    try:
        lines = response.iter_lines(decode_unicode=True)
        for raw_line in lines:
            line = (
                raw_line.decode("utf-8", errors="replace")
                if isinstance(raw_line, bytes)
                else raw_line
            )
            if not isinstance(line, str) or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            chunk = json.loads(data)
            choices = chunk.get("choices", [])
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta", {})
            if not isinstance(delta, Mapping):
                continue
            reasoning = _delta_text(
                delta.get("reasoning_content", delta.get("reasoning"))
            )
            content = _delta_text(delta.get("content"))
            if reasoning:
                reasoning_parts.append(reasoning)
                _terminal_chunk(reasoning, thinking=True)
            if content:
                content_parts.append(content)
                _terminal_chunk(content, thinking=False)
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise StructuredLlmError(f"{operation}: invalid vLLM stream: {error}") from error
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    return _CompletionMessage("".join(content_parts), "".join(reasoning_parts), finish_reason)


def _buffered_completion(response: requests.Response, operation: str) -> _CompletionMessage:
    try:
        envelope = response.json()
        choice = envelope["choices"][0]
        message = choice["message"]
        content = message.get("content")
        reasoning = message.get("reasoning_content", message.get("reasoning"))
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise StructuredLlmError(f"{operation}: invalid vLLM response envelope: {error}") from error
    content_text = content if isinstance(content, str) else ""
    reasoning_text = reasoning if isinstance(reasoning, str) else ""
    embedded_reasoning, content_text = _split_embedded_thinking(content_text)
    reasoning_text = f"{reasoning_text}{embedded_reasoning}"
    if reasoning_text:
        _terminal_chunk(reasoning_text, thinking=True)
    if content_text:
        _terminal_chunk(content_text, thinking=False)
    return _CompletionMessage(
        content=content_text,
        reasoning_content=reasoning_text,
        finish_reason=str(finish_reason) if finish_reason is not None else None,
    )


def _split_embedded_thinking(content: str) -> tuple[str, str]:
    matches = tuple(re.finditer(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE))
    if not matches:
        return "", content
    reasoning = "\n".join(match.group(1).strip() for match in matches if match.group(1).strip())
    visible = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    return reasoning, visible.strip()


def _delta_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "".join(
        str(item.get("text", ""))
        for item in value
        if isinstance(item, Mapping) and item.get("type") == "text"
    )


def _terminal_header(operation: str) -> None:
    label = _OPERATION_LABELS.get(operation, operation.upper())
    sys.stdout.write(f"\n{_ANSI_HEADER}━━ {label} ━━{_ANSI_RESET}\n")
    sys.stdout.flush()


def _terminal_chunk(value: str, *, thinking: bool) -> None:
    if not value:
        return
    style = _ANSI_THINKING if thinking else _ANSI_RESET
    sys.stdout.write(f"{style}{value}{_ANSI_RESET}")
    sys.stdout.flush()


def _terminal_success(response: requests.Response, operation: str, content_length: int) -> None:
    started_at = getattr(response, "_frod_started_at", time.monotonic())
    duration = max(0.0, time.monotonic() - float(started_at))
    label = _OPERATION_LABELS.get(operation, operation)
    sys.stdout.write(
        f"\n{_ANSI_SUCCESS}✓ {label} terminé en {duration:.2f} s · "
        f"{content_length} caractères{_ANSI_RESET}\n"
    )
    sys.stdout.flush()


def _terminal_rejected(response: requests.Response, message: str) -> None:
    sys.stdout.write(f"{_ANSI_WARNING}↻ {message}{_ANSI_RESET}\n")
    sys.stdout.flush()
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _response_object(response: requests.Response, operation: str) -> _DecodedResponse:
    try:
        response.raise_for_status()
        completion = _completion_message(response, operation)
        content = completion.content
        finish_reason = completion.finish_reason
    except (TypeError, ValueError, requests.RequestException) as error:
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
