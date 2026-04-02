#!/usr/bin/env python3
"""Anthropic-to-OpenAI compatibility proxy for Claude Code.

This server exposes a minimal subset of the Anthropic Messages API that
Claude Code expects and forwards requests to an OpenAI-compatible
`/v1/chat/completions` backend.

Supported endpoints:
- POST /v1/messages
- POST /v1/messages/count_tokens
- GET  /healthz

Environment:
- PROXY_HOST: bind host, default 127.0.0.1
- PROXY_PORT: bind port, default 4000
- OPENAI_BASE_URL: upstream base URL, default https://api.openai.com/v1
- OPENAI_API_KEY: upstream API key
- OPENAI_TIMEOUT: upstream request timeout in seconds, default 600
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


PROXY_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "4000"))
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "600"))
SERVER_NAME = "claude-openai-proxy/0.1"


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def utc_now() -> int:
    return int(time.time())


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    # This is intentionally conservative and only used for count_tokens.
    return max(1, math.ceil(len(text) / 4))


def anthropic_block_text(block: dict[str, Any]) -> str:
    block_type = block.get("type")
    if block_type == "text":
        return block.get("text", "")
    if block_type == "tool_result":
        content = block.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(anthropic_block_text(item) for item in content if isinstance(item, dict))
        return json_dumps(content)
    if block_type == "tool_use":
        return json_dumps(
            {
                "tool_use_id": block.get("id"),
                "name": block.get("name"),
                "input": block.get("input", {}),
            }
        )
    return json_dumps(block)


def anthropic_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(anthropic_block_text(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def normalize_system_prompt(system: Any) -> str | None:
    text = anthropic_content_to_text(system).strip()
    return text or None


def map_tool_choice(tool_choice: Any) -> Any:
    if not tool_choice:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        return None
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "tool" and tool_choice.get("name"):
            return {"type": "function", "function": {"name": tool_choice["name"]}}
    return None


def anthropic_tools_to_openai(tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tools, list) or not tools:
        return None

    openai_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return openai_tools or None


def anthropic_messages_to_openai(messages: list[dict[str, Any]], system: Any) -> list[dict[str, Any]]:
    openai_messages: list[dict[str, Any]] = []
    system_prompt = normalize_system_prompt(system)
    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role not in {"user", "assistant"}:
            continue
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            openai_messages.append({"role": role, "content": str(content)})
            continue

        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue

            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    text_parts.append(text)
                continue

            if block_type == "tool_use" and role == "assistant":
                if text_parts:
                    openai_messages.append({"role": "assistant", "content": "\n".join(text_parts)})
                    text_parts = []
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": block.get("id") or f"toolu_{uuid.uuid4().hex}",
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json_dumps(block.get("input", {})),
                                },
                            }
                        ],
                    }
                )
                continue

            if block_type == "tool_result" and role == "user":
                if text_parts:
                    openai_messages.append({"role": "user", "content": "\n".join(text_parts)})
                    text_parts = []
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") or block.get("id", ""),
                        "content": anthropic_content_to_text(block.get("content", "")),
                    }
                )
                continue

            text_parts.append(anthropic_block_text(block))

        if text_parts:
            openai_messages.append({"role": role, "content": "\n".join(text_parts)})

    return openai_messages


def build_openai_request(payload: dict[str, Any], stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": payload["model"],
        "messages": anthropic_messages_to_openai(payload.get("messages", []), payload.get("system")),
        "stream": stream,
    }

    if payload.get("max_tokens") is not None:
        body["max_completion_tokens"] = payload["max_tokens"]
    if payload.get("temperature") is not None:
        body["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        body["top_p"] = payload["top_p"]
    if payload.get("stop_sequences"):
        body["stop"] = payload["stop_sequences"]

    tools = anthropic_tools_to_openai(payload.get("tools"))
    if tools:
        body["tools"] = tools
        tool_choice = map_tool_choice(payload.get("tool_choice"))
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

    if stream:
        body["stream_options"] = {"include_usage": True}

    return body


def post_upstream_json(path: str, body: dict[str, Any]) -> dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    req = urllib.request.Request(
        f"{OPENAI_BASE_URL}{path}",
        data=json_dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OPENAI_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise UpstreamError.from_http_error(exc) from exc
    except urllib.error.URLError as exc:
        raise UpstreamError(502, f"upstream connection failed: {exc.reason}") from exc


@dataclass
class UpstreamError(Exception):
    status_code: int
    message: str
    error_type: str = "upstream_error"
    raw_body: Any = None

    @classmethod
    def from_http_error(cls, exc: urllib.error.HTTPError) -> "UpstreamError":
        body_text = exc.read().decode("utf-8", errors="replace")
        message = body_text
        error_type = "upstream_error"
        raw_body: Any = body_text
        try:
            body_json = json.loads(body_text)
            raw_body = body_json
            if isinstance(body_json, dict):
                error = body_json.get("error", body_json)
                if isinstance(error, dict):
                    message = error.get("message", message)
                    error_type = error.get("type", error_type)
        except json.JSONDecodeError:
            pass
        return cls(exc.code, message, error_type=error_type, raw_body=raw_body)


def anthropic_usage_from_openai(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
    }


def anthropic_response_from_openai(payload: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    choices = upstream.get("choices") or []
    if not choices:
        raise UpstreamError(502, "upstream returned no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []

    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            parsed_args = json.loads(arguments)
        except json.JSONDecodeError:
            parsed_args = {"raw_arguments": arguments}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex}",
                "name": function.get("name", ""),
                "input": parsed_args,
            }
        )

    finish_reason = choice.get("finish_reason")
    stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"

    return {
        "id": upstream.get("id", f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": payload["model"],
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage_from_openai(upstream.get("usage")),
    }


@dataclass
class StreamBlockState:
    index: int
    kind: str
    tool_id: str | None = None
    tool_name: str | None = None
    tool_args_parts: list[str] = field(default_factory=list)
    started: bool = False


class ClaudeOpenAIProxyHandler(BaseHTTPRequestHandler):
    server_version = SERVER_NAME
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json_dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str, error_type: str = "invalid_request_error") -> None:
        self._send_json(status, {"type": "error", "error": {"type": error_type, "message": message}})

    def _send_sse_event(self, event: str, data: dict[str, Any]) -> None:
        payload = f"event: {event}\ndata: {json_dumps(data)}\n\n".encode("utf-8")
        self.wfile.write(payload)
        self.wfile.flush()

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(
                200,
                {
                    "ok": True,
                    "server": SERVER_NAME,
                    "openai_base_url": OPENAI_BASE_URL,
                    "has_openai_api_key": bool(OPENAI_API_KEY),
                },
            )
            return
        self._send_error_json(404, f"unknown path: {self.path}", "not_found_error")

    def do_POST(self) -> None:
        try:
            if self.path == "/v1/messages":
                self._handle_messages()
                return
            if self.path == "/v1/messages/count_tokens":
                self._handle_count_tokens()
                return
            self._send_error_json(404, f"unknown path: {self.path}", "not_found_error")
        except BrokenPipeError:
            pass
        except UpstreamError as exc:
            self._send_error_json(exc.status_code, exc.message, exc.error_type)
        except ValueError as exc:
            self._send_error_json(400, str(exc))
        except Exception as exc:  # pragma: no cover - defensive path
            traceback.print_exc()
            self._send_error_json(500, f"proxy internal error: {exc}", "internal_error")

    def _handle_count_tokens(self) -> None:
        payload = self._read_json()
        serialized = {
            "system": payload.get("system"),
            "messages": payload.get("messages", []),
            "tools": payload.get("tools", []),
        }
        estimate = estimate_tokens_from_text(json_dumps(serialized))
        self._send_json(200, {"input_tokens": estimate})

    def _handle_messages(self) -> None:
        payload = self._read_json()
        if not payload.get("model"):
            raise ValueError("model is required")

        stream = bool(payload.get("stream"))
        if stream:
            self._handle_messages_stream(payload)
            return

        upstream = post_upstream_json("/chat/completions", build_openai_request(payload, stream=False))
        self._send_json(200, anthropic_response_from_openai(payload, upstream))

    def _handle_messages_stream(self, payload: dict[str, Any]) -> None:
        body = build_openai_request(payload, stream=True)
        req = urllib.request.Request(
            f"{OPENAI_BASE_URL}/chat/completions",
            data=json_dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            method="POST",
        )

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        message_id = f"msg_{uuid.uuid4().hex}"
        input_tokens = 0
        output_tokens = 0
        text_block_started = False
        text_block_index = -1
        tool_blocks: dict[int, StreamBlockState] = {}
        next_content_index = 0
        stop_reason = "end_turn"

        self._send_sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": payload["model"],
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=OPENAI_TIMEOUT) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    usage = chunk.get("usage") or {}
                    if usage:
                        input_tokens = int(usage.get("prompt_tokens", input_tokens) or input_tokens)
                        output_tokens = int(usage.get("completion_tokens", output_tokens) or output_tokens)
                    if not choices:
                        continue

                    delta = choices[0].get("delta") or {}
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason == "tool_calls":
                        stop_reason = "tool_use"
                    elif finish_reason and finish_reason != "stop":
                        stop_reason = finish_reason

                    text_delta = delta.get("content")
                    if text_delta:
                        if not text_block_started:
                            text_block_index = next_content_index
                            self._send_sse_event(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": text_block_index,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )
                            text_block_started = True
                            next_content_index += 1
                        self._send_sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": text_block_index,
                                "delta": {"type": "text_delta", "text": text_delta},
                            },
                        )

                    for tool_delta in delta.get("tool_calls") or []:
                        tool_idx = int(tool_delta.get("index", len(tool_blocks)))
                        state = tool_blocks.get(tool_idx)
                        if state is None:
                            state = StreamBlockState(index=next_content_index, kind="tool_use")
                            tool_blocks[tool_idx] = state
                            next_content_index += 1

                        if tool_delta.get("id"):
                            state.tool_id = tool_delta["id"]
                        function = tool_delta.get("function") or {}
                        if function.get("name"):
                            state.tool_name = function["name"]

                        if state.tool_id and state.tool_name and not state.started:
                            self._send_sse_event(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": state.index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": state.tool_id,
                                        "name": state.tool_name,
                                        "input": {},
                                    },
                                },
                            )
                            state.started = True

                        arguments_delta = function.get("arguments")
                        if arguments_delta:
                            if not state.tool_id:
                                state.tool_id = f"toolu_{uuid.uuid4().hex}"
                            if not state.tool_name:
                                state.tool_name = "unknown"
                            if not state.started:
                                self._send_sse_event(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": state.index,
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": state.tool_id,
                                            "name": state.tool_name,
                                            "input": {},
                                        },
                                    },
                                )
                                state.started = True
                            state.tool_args_parts.append(arguments_delta)
                            self._send_sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": state.index,
                                    "delta": {"type": "input_json_delta", "partial_json": arguments_delta},
                                },
                            )

                if text_block_started:
                    self._send_sse_event(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": text_block_index},
                    )
                for state in tool_blocks.values():
                    if state.started:
                        self._send_sse_event("content_block_stop", {"type": "content_block_stop", "index": state.index})

                self._send_sse_event(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": {"output_tokens": output_tokens},
                    },
                )
                self._send_sse_event("message_stop", {"type": "message_stop"})
        except urllib.error.HTTPError as exc:
            upstream_err = UpstreamError.from_http_error(exc)
            self._send_sse_event(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": upstream_err.error_type,
                        "message": upstream_err.message,
                    },
                },
            )
        except urllib.error.URLError as exc:
            self._send_sse_event(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "upstream_connection_error",
                        "message": f"upstream connection failed: {exc.reason}",
                    },
                },
            )


def run_server() -> None:
    server = ThreadingHTTPServer((PROXY_HOST, PROXY_PORT), ClaudeOpenAIProxyHandler)
    print(
        f"{SERVER_NAME} listening on http://{PROXY_HOST}:{PROXY_PORT} "
        f"-> {OPENAI_BASE_URL}/chat/completions",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    run_server()
