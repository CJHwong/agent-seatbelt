#!/usr/bin/env python3
"""A scripted Anthropic API, so a hook control can be tested without a model.

Driving a control through a real model is not a test. The model decides whether to
call a tool at all, and it will refuse a prompt that reads like a probe, so a missing
side effect proves nothing. This serves one fixed tool call to the conversation and
then ends the turn, which leaves the hook as the only variable.

    stub_anthropic_api.py --port N --command "touch /tmp/sentinel"

    ANTHROPIC_BASE_URL=http://127.0.0.1:N ANTHROPIC_API_KEY=stub \
      ANTHROPIC_MODEL=claude-sonnet-5 claude -p "anything"

GET / reports how many requests arrived and whether any carried the marker, which is
how a canary tells "the control blocked the call" from "the call never happened".
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

STATE: dict[str, Any] = {
    "requests": 0,
    "tool": "Bash",
    "command": "true",
    "marker": "",
    "saw_marker": False,
    "issued_tool": False,
    "detail": [],
}


def _digest(raw: str, body: dict) -> dict:
    """What one request carried, so a canary can name the leak instead of the count.

    The CLI makes more than one call per turn (a session-title call is separate), so
    "a request carried the marker" is not the same as "the conversation ran". This
    keeps enough to tell those apart.
    """
    messages = body.get("messages") or []
    seen = []
    for message in messages:
        blocks = message.get("content")
        text = (
            blocks
            if isinstance(blocks, str)
            else "".join(
                block.get("text", "")
                for block in blocks or []
                if isinstance(block, dict)
            )
        )
        seen.append(
            {
                "role": message.get("role"),
                "carries_marker": bool(STATE["marker"]) and STATE["marker"] in text,
                "head": text[:60],
            }
        )
    return {
        "carries_marker": bool(STATE["marker"]) and STATE["marker"] in raw,
        "max_tokens": body.get("max_tokens"),
        "tools": len(body.get("tools") or []),
        "stream": bool(body.get("stream")),
        "messages": seen[:4],
    }


def _message_start(model: str) -> dict:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_stub",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }


def text_events(text: str, model: str) -> list[tuple[str, dict]]:
    return [
        ("message_start", _message_start(model)),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


def tool_events(model: str) -> list[tuple[str, dict]]:
    arguments = json.dumps({"command": STATE["command"]})
    return [
        ("message_start", _message_start(model)),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_stub",
                    "name": STATE["tool"],
                    "input": {},
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": arguments},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


def assemble(events: list[tuple[str, dict]]) -> dict:
    content: list[dict] = []
    stop_reason = "end_turn"
    for name, payload in events:
        if name == "content_block_start":
            content.append(payload["content_block"])
        elif name == "content_block_delta":
            delta = payload["delta"]
            if delta["type"] == "text_delta":
                content[-1]["text"] = content[-1].get("text", "") + delta["text"]
            else:
                content[-1]["input"] = json.loads(delta["partial_json"])
        elif name == "message_delta":
            stop_reason = payload["delta"]["stop_reason"]
    return {
        "id": "msg_stub",
        "type": "message",
        "role": "assistant",
        "model": "stub-model",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", "replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {}

        if "count_tokens" in self.path:
            self._send(json.dumps({"input_tokens": 1}), "application/json")
            return

        STATE["requests"] += 1
        if STATE["marker"] and STATE["marker"] in raw:
            STATE["saw_marker"] = True
        STATE["detail"].append(_digest(raw, body))
        # A session-title call arrives before the turn and carries no tools. Keying
        # the scripted call on "request number one" put it on that call, which the
        # CLI discards, so the conversation saw no tool call and a working control
        # read as a block. A request that carries tools is a conversation request.
        if body.get("tools") and not STATE["issued_tool"]:
            STATE["issued_tool"] = True
            events = tool_events(body.get("model", "stub-model"))
        else:
            events = text_events("done", body.get("model", "stub-model"))
        if body.get("stream"):
            payload = "".join(
                f"event: {n}\ndata: {json.dumps(b)}\n\n" for n, b in events
            )
            self._send(payload, "text/event-stream")
        else:
            self._send(json.dumps(assemble(events)), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        self._send(
            json.dumps(
                {
                    "requests": STATE["requests"],
                    "saw_marker": STATE["saw_marker"],
                    "detail": STATE["detail"],
                }
            ),
            "application/json",
        )

    def _send(self, body: str, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tool", default="Bash")
    parser.add_argument("--command", default="true")
    # A marker the canary puts in its prompt. The CLI may call the API for a session
    # title even when the prompt hook blocks, so counting requests cannot distinguish
    # "the prompt never arrived" from "something else did". Watching for the marker can.
    parser.add_argument("--marker", default="")
    args = parser.parse_args()
    STATE["tool"] = args.tool
    STATE["command"] = args.command
    STATE["marker"] = args.marker
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(
        f"stub on {args.port}: {args.tool} {args.command}", file=sys.stderr, flush=True
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
