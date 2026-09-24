from __future__ import annotations

import contextlib
import http.client
import importlib.util
import io
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import types
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest import TestCase
from unittest.mock import patch


HOOKS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOOKS_DIR))

MODULE_PATH = HOOKS_DIR / "pii-server.py"
MODULE_SPEC = importlib.util.spec_from_file_location("pii_server", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot load server module: {MODULE_PATH}")
PII_SERVER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(PII_SERVER)


class PiiServerModeTests(TestCase):
    def test_default_mode_is_redact(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(PII_SERVER.resolve_mode(), "redact")

    def test_environment_selects_openai_mode(self) -> None:
        with patch.dict(os.environ, {"PII_SERVER_MODE": "openai"}, clear=True):
            self.assertEqual(PII_SERVER.resolve_mode(), "openai")

    def test_argument_overrides_environment(self) -> None:
        with patch.dict(os.environ, {"PII_SERVER_MODE": "openai"}, clear=True):
            self.assertEqual(PII_SERVER.resolve_mode("redact"), "redact")

    def test_invalid_mode_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "PII_SERVER_MODE"):
            PII_SERVER.resolve_mode("unknown")

    def test_health_payload_identifies_mode_and_device(self) -> None:
        model = type("Model", (), {"device": "mps"})()
        self.assertEqual(
            PII_SERVER.health_payload("redact", model),
            {"status": "ok", "mode": "redact", "device": "mps", "busy": False},
        )
        self.assertEqual(
            PII_SERVER.health_payload("openai", model),
            {"status": "ok", "mode": "openai", "device": "cpu", "busy": False},
        )

    def test_health_payload_carries_the_busy_flag(self) -> None:
        self.assertEqual(
            PII_SERVER.health_payload("rules", PII_SERVER.RulesModel(), busy=True),
            {"status": "ok", "mode": "rules", "device": "cpu", "busy": True},
        )

    def test_environment_selects_rules_mode(self) -> None:
        with patch.dict(os.environ, {"PII_SERVER_MODE": "rules"}, clear=True):
            self.assertEqual(PII_SERVER.resolve_mode(), "rules")

    def test_rules_mode_reports_no_accelerator(self) -> None:
        self.assertEqual(
            PII_SERVER.health_payload("rules", PII_SERVER.RulesModel()),
            {"status": "ok", "mode": "rules", "device": "cpu", "busy": False},
        )


class BodyCapTests(TestCase):
    def test_default_cap_is_two_mebibytes(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(PII_SERVER.max_body_bytes(), 2 * 1024 * 1024)

    def test_environment_raises_the_cap(self) -> None:
        with patch.dict(os.environ, {"PII_MAX_BODY_BYTES": "4096"}):
            self.assertEqual(PII_SERVER.max_body_bytes(), 4096)

    def test_malformed_setting_falls_back_to_the_default(self) -> None:
        with patch.dict(os.environ, {"PII_MAX_BODY_BYTES": "not-a-number"}):
            self.assertEqual(PII_SERVER.max_body_bytes(), 2 * 1024 * 1024)

    def test_non_positive_setting_falls_back_to_the_default(self) -> None:
        with patch.dict(os.environ, {"PII_MAX_BODY_BYTES": "0"}):
            self.assertEqual(PII_SERVER.max_body_bytes(), 2 * 1024 * 1024)


class RulesModelTests(TestCase):
    def test_empty_input_returns_no_spans(self) -> None:
        self.assertEqual(PII_SERVER.RulesModel().predict(""), [])

    def test_secret_is_found_without_a_checkpoint(self) -> None:
        spans = PII_SERVER.RulesModel().predict(
            "Staging key sk_live_9Kx3Lm2Qp7Yt4Nf8Rw1Zc6Vb sits in .env."
        )
        self.assertEqual([span["label"] for span in spans], ["secret"])

    def test_email_and_phone_are_found(self) -> None:
        spans = PII_SERVER.RulesModel().predict(
            "Ping dana.reyes@example.org or +1 (415) 555-0134."
        )
        self.assertEqual(
            sorted(span["label"] for span in spans),
            ["private_email", "private_phone"],
        )

    def test_person_name_is_not_found(self) -> None:
        self.assertEqual(
            PII_SERVER.RulesModel().predict("Add Avery Coleman to the invite."),
            [],
        )

    def test_rules_module_pulls_no_model_framework(self) -> None:
        probe = (
            "import sys; sys.path.insert(0, %r); import pii_rules; "
            "print(sorted(m for m in ('torch', 'numpy', 'onnxruntime', 'tokenizers') "
            "if m in sys.modules))" % str(HOOKS_DIR)
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "[]")

    def test_server_module_pulls_no_model_framework(self) -> None:
        probe = (
            "import importlib.util, sys; sys.path.insert(0, %r); "
            "spec = importlib.util.spec_from_file_location('pii_server', %r); "
            "module = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            "print(sorted(m for m in ('torch', 'numpy', 'onnxruntime', 'tokenizers') "
            "if m in sys.modules))" % (str(HOOKS_DIR), str(MODULE_PATH))
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "[]")

    def test_rules_mode_stays_free_of_the_model_frameworks(self) -> None:
        probe = (
            "import importlib.util, sys; sys.path.insert(0, %r); "
            "spec = importlib.util.spec_from_file_location('pii_server', %r); "
            "module = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            "module.load_selected_model('rules').predict('Ping dana@example.org.'); "
            "print(sorted(m for m in ('torch', 'numpy', 'onnxruntime', 'tokenizers') "
            "if m in sys.modules))" % (str(HOOKS_DIR), str(MODULE_PATH))
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "[]")


class StubRedactModel:
    """Records what pii-server forwards to the Redact constructor."""

    created: ClassVar[list[StubRedactModel]] = []

    def __init__(self, cache_dir: Path, requested_device: str | None) -> None:
        self.cache_dir = cache_dir
        self.requested_device = requested_device
        StubRedactModel.created.append(self)


ASSET_DIRECTORY = Path("/nonexistent/redact-assets")


@contextlib.contextmanager
def stubbed_redact_module() -> Iterator[None]:
    """Swap in a Redact module, so the test needs no 88 MB checkpoint."""
    stub_module = types.ModuleType("pii_redact_torch")
    setattr(stub_module, "RedactModel", StubRedactModel)
    setattr(stub_module, "ensure_assets", lambda: ASSET_DIRECTORY)
    StubRedactModel.created = []
    with patch.dict(sys.modules, {"pii_redact_torch": stub_module}):
        yield


class StubLiteRedactModel:
    """Records what pii-server forwards to the LiteRT constructor."""

    created: ClassVar[list[StubLiteRedactModel]] = []

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        StubLiteRedactModel.created.append(self)


@contextlib.contextmanager
def stubbed_lite_module() -> Iterator[None]:
    """Swap in the LiteRT backend, so the test needs no 23 MB graph."""
    stub_module = types.ModuleType("pii_redact_lite")
    setattr(stub_module, "LitertRedactModel", StubLiteRedactModel)
    setattr(stub_module, "ensure_assets", lambda: ASSET_DIRECTORY)
    StubLiteRedactModel.created = []
    with patch.dict(sys.modules, {"pii_redact_lite": stub_module}):
        yield


class ModelSelectionTests(TestCase):
    def test_openai_mode_builds_the_model_from_the_asset_directory(self) -> None:
        stub_module = types.ModuleType("pii_opf")
        built: list[Path] = []

        class StubModel:
            def __init__(self, cache_dir: Path) -> None:
                built.append(cache_dir)

        setattr(stub_module, "Model", StubModel)
        setattr(stub_module, "ensure_assets", lambda: ASSET_DIRECTORY)
        with patch.dict(sys.modules, {"pii_opf": stub_module}):
            model = PII_SERVER.load_selected_model("openai")

        self.assertIsInstance(model, StubModel)
        self.assertEqual(built, [ASSET_DIRECTORY])

    def test_redact_mode_builds_the_litert_model_from_the_asset_directory(self) -> None:
        """`redact` is what a fresh install selects, so it has to be the backend
        that fetches its own assets rather than the one that needs a checkpoint."""
        with stubbed_lite_module():
            model = PII_SERVER.load_selected_model("redact")

        self.assertIsInstance(model, StubLiteRedactModel)
        self.assertEqual(StubLiteRedactModel.created[0].cache_dir, ASSET_DIRECTORY)

    def test_pii_redact_torch_mode_defaults_the_requested_device_to_auto(self) -> None:
        with stubbed_redact_module(), patch.dict(os.environ, {}, clear=True):
            model = PII_SERVER.load_selected_model("redact-torch")

        self.assertIs(model, StubRedactModel.created[0])
        self.assertEqual(StubRedactModel.created[0].cache_dir, ASSET_DIRECTORY)
        self.assertEqual(StubRedactModel.created[0].requested_device, "auto")

    def test_pii_redact_torch_mode_forwards_the_requested_device(self) -> None:
        with (
            stubbed_redact_module(),
            patch.dict(os.environ, {"REDACT_DEVICE": "cpu"}, clear=True),
        ):
            PII_SERVER.load_selected_model("redact-torch")

        self.assertEqual(StubRedactModel.created[0].requested_device, "cpu")

    def test_server_directory_is_added_to_the_import_path(self) -> None:
        server_directory = str(HOOKS_DIR)
        without_server_directory = [
            entry for entry in sys.path if entry != server_directory
        ]
        self.assertNotIn(server_directory, without_server_directory)

        with patch.object(sys, "path", without_server_directory):
            model = PII_SERVER.load_selected_model("rules")
            self.assertIn(server_directory, sys.path)

        self.assertIsInstance(model, PII_SERVER.RulesModel)

    def test_server_directory_is_not_added_when_it_is_already_there(self) -> None:
        server_directory = str(HOOKS_DIR)
        present_already = sys.path.count(server_directory)
        self.assertGreater(present_already, 0)

        with patch.object(sys, "path", [server_directory, *sys.path]):
            PII_SERVER.load_selected_model("rules")
            self.assertEqual(sys.path.count(server_directory), present_already + 1)


def free_port() -> int:
    """A port nothing is listening on. Racy by nature; adequate for a test."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def request_json(
    port: int, method: str, path: str, body: bytes | None = None
) -> tuple[int, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        connection.request(method, path, body=body)
        response = connection.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()


def declared_length_request(port: int, content_length: str) -> tuple[int, Any]:
    """Declare a Content-Length, send no body, and read the answer.

    A handler that reads the declared body before it answers blocks here, so any
    reply at all proves the status came back ahead of the read.
    """
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.putrequest("POST", "/")
        connection.putheader("Content-Length", content_length)
        connection.endheaders()
        response = connection.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()


def wait_for_health(port: int, timeout: float = 20.0) -> tuple[int, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, payload = request_json(port, "GET", "/health")
        except OSError:
            time.sleep(0.05)
            continue
        if status == 200:
            return status, payload
        time.sleep(0.05)
    raise AssertionError(f"the server on port {port} never answered /health")


class HandlerRequestTests(TestCase):
    """Drive the real handler over a real socket on an ephemeral port."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), PII_SERVER.Handler)
        cls.port = int(cls.server.server_address[1])
        cls.serving_thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.serving_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.serving_thread.join(timeout=2)

    def setUp(self) -> None:
        PII_SERVER.Handler.mode = "rules"
        PII_SERVER.Handler.model = PII_SERVER.RulesModel()
        self.server_log = io.StringIO()
        redirect = contextlib.redirect_stderr(self.server_log)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def request(self, method: str, path: str, body: bytes | None = None):
        return request_json(self.port, method, path, body)

    def post_text(self, text: str):
        return self.request("POST", "/", json.dumps({"text": text}).encode("utf-8"))

    def test_health_endpoint_reports_the_rules_mode(self) -> None:
        self.assertEqual(
            self.request("GET", "/health"),
            (200, {"status": "ok", "mode": "rules", "device": "cpu", "busy": False}),
        )

    def test_unknown_path_is_not_found(self) -> None:
        self.assertEqual(self.request("GET", "/spans"), (404, {"error": "not found"}))

    def test_each_request_is_logged_with_its_mode(self) -> None:
        self.request("GET", "/health")

        self.assertEqual(
            self.server_log.getvalue().strip(),
            '[rules] 127.0.0.1 "GET /health HTTP/1.1" 200 -',
        )

    def test_empty_body_is_rejected(self) -> None:
        self.assertEqual(self.request("POST", "/", b""), (400, {"error": "empty body"}))

    def test_malformed_json_is_rejected(self) -> None:
        status, payload = self.request("POST", "/", b"{not json")

        self.assertEqual(status, 400)
        self.assertTrue(payload["error"].startswith("bad request: "), payload)

    def test_missing_text_field_is_rejected(self) -> None:
        status, payload = self.request("POST", "/", b'{"other": "value"}')

        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "bad request: 'text'")

    def test_non_string_text_is_rejected(self) -> None:
        self.assertEqual(
            self.request("POST", "/", b'{"text": 42}'),
            (400, {"error": "bad request: text must be a string"}),
        )

    def test_non_object_json_body_is_rejected(self) -> None:
        for body in (b'"hello"', b"[1,2,3]", b"null", b"42", b"true"):
            with self.subTest(body=body):
                self.assertEqual(
                    self.request("POST", "/", body),
                    (400, {"error": "bad request: body must be a JSON object"}),
                )

    def test_non_numeric_content_length_is_rejected(self) -> None:
        self.assertEqual(
            declared_length_request(self.port, "abc"),
            (400, {"error": "invalid Content-Length"}),
        )

    def test_declared_body_above_the_cap_is_answered_before_the_read(self) -> None:
        with patch.dict(os.environ, {"PII_MAX_BODY_BYTES": "64"}):
            self.assertEqual(
                declared_length_request(self.port, "65"),
                (
                    413,
                    {"error": "request body of 65 bytes exceeds the 64 byte limit"},
                ),
            )

    def test_body_at_the_cap_is_read_and_judged_on_its_content(self) -> None:
        with patch.dict(os.environ, {"PII_MAX_BODY_BYTES": "64"}):
            status, payload = self.request("POST", "/", b"x" * 64)

        self.assertEqual(status, 400)
        self.assertTrue(payload["error"].startswith("bad request: "), payload)

    def test_health_reports_busy_while_a_request_holds_the_lock(self) -> None:
        holding = threading.Event()
        release = threading.Event()

        class BlockingModel:
            device = "cpu"

            def predict(self, text: str) -> list[dict]:
                holding.set()
                release.wait(timeout=10)
                return []

        PII_SERVER.Handler.model = BlockingModel()
        caller = threading.Thread(
            target=self.post_text, args=("anything",), daemon=True
        )
        caller.start()
        try:
            self.assertTrue(holding.wait(timeout=5))
            self.assertEqual(
                self.request("GET", "/health"),
                (
                    200,
                    {"status": "ok", "mode": "rules", "device": "cpu", "busy": True},
                ),
            )
        finally:
            release.set()
            caller.join(timeout=10)

        self.assertFalse(self.request("GET", "/health")[1]["busy"])

    def test_a_stalled_connection_is_dropped_by_the_handler_timeout(self) -> None:
        with patch.object(PII_SERVER.Handler, "timeout", 0.3):
            connection = socket.create_connection(("127.0.0.1", self.port), timeout=10)
            try:
                connection.sendall(
                    b"POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 50\r\n\r\n"
                )
                started = time.monotonic()
                self.assertEqual(connection.recv(1024), b"")
                elapsed = time.monotonic() - started
            finally:
                connection.close()

        self.assertLess(elapsed, 5)

    def test_model_rejection_becomes_request_entity_too_large(self) -> None:
        class RejectingModel:
            device = "cpu"

            def predict(self, text: str) -> list[dict]:
                raise ValueError("input exceeds max 4 tokens")

        PII_SERVER.Handler.model = RejectingModel()
        self.assertEqual(
            self.post_text("a b c d e"),
            (413, {"error": "input exceeds max 4 tokens"}),
        )

    def test_empty_text_is_not_the_same_as_an_empty_body(self) -> None:
        status, payload = self.post_text("")

        self.assertEqual(status, 200)
        self.assertEqual(payload["spans"], [])

    def test_concurrent_posts_get_their_own_answer(self) -> None:
        class SlowModel:
            device = "cpu"

            def __init__(self) -> None:
                self.active = 0
                self.peak = 0
                self.guard = threading.Lock()

            def predict(self, text: str) -> list[dict]:
                with self.guard:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(0.1)
                with self.guard:
                    self.active -= 1
                return [{"start": 0, "end": len(text), "label": "echo", "text": text}]

        slow_model = SlowModel()
        PII_SERVER.Handler.model = slow_model
        answers: list[tuple[int, int, list[str]]] = []

        def ask(index: int) -> None:
            status, payload = self.post_text(f"request {index}")
            answers.append((index, status, [span["text"] for span in payload["spans"]]))

        callers = [threading.Thread(target=ask, args=(index,)) for index in range(4)]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join()

        self.assertEqual(slow_model.peak, 1)
        self.assertEqual(
            sorted(answers),
            [(index, 200, [f"request {index}"]) for index in range(4)],
        )

    def test_lock_is_released_when_the_model_raises(self) -> None:
        class ExplodingModel:
            device = "cpu"

            def predict(self, text: str) -> list[dict]:
                raise ValueError("model exploded")

        PII_SERVER.Handler.model = ExplodingModel()
        for _ in range(3):
            self.assertEqual(
                self.post_text("anything"),
                (413, {"error": "model exploded"}),
            )

        PII_SERVER.Handler.model = PII_SERVER.RulesModel()
        self.assertEqual(self.post_text("nothing sensitive here")[0], 200)
        self.assertFalse(PII_SERVER.Handler.inference_lock.locked())

    def test_a_model_shape_failure_becomes_internal_server_error(self) -> None:
        shape_error = "model output shape (4, 13) does not match 13 labels"

        class BrokenModel:
            device = "cpu"

            def predict(self, text: str) -> list[dict]:
                raise RuntimeError(shape_error)

        PII_SERVER.Handler.model = BrokenModel()
        self.assertEqual(self.post_text("anything"), (500, {"error": shape_error}))
        self.assertIn(shape_error, self.server_log.getvalue())

        PII_SERVER.Handler.model = PII_SERVER.RulesModel()
        self.assertEqual(self.post_text("nothing sensitive here")[0], 200)

    def test_a_client_that_leaves_before_the_answer_is_logged_not_traced(self) -> None:
        """A swallowed disconnect handler fails silently, so drive a real one.

        The client hangs up between the request and the answer, which is the
        ordinary case the handler exists for.
        """
        reached_model = threading.Event()
        release_model = threading.Event()

        class SignallingModel:
            device = "cpu"

            def predict(self, text: str) -> list[dict]:
                reached_model.set()
                release_model.wait(timeout=10)
                return []

        PII_SERVER.Handler.model = SignallingModel()
        connection = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        try:
            body = json.dumps({"text": "nothing sensitive here"}).encode("utf-8")
            connection.sendall(
                b"POST / HTTP/1.0\r\nHost: 127.0.0.1\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            self.assertTrue(
                reached_model.wait(timeout=5), "the request never reached the model"
            )

            # Leave without reading the answer. SO_LINGER 0 sends a reset, so
            # the server's write fails rather than racing the close.
            connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            connection.close()
            release_model.set()

            self.assertTrue(
                wait_until(
                    lambda: "client disconnected before the response"
                    in self.server_log.getvalue()
                ),
                self.server_log.getvalue(),
            )
        finally:
            release_model.set()
            with contextlib.suppress(OSError):
                connection.close()

        self.assertNotIn("Traceback", self.server_log.getvalue())
        self.assertEqual(self.post_text("nothing sensitive here")[0], 200)

    def test_post_returns_spans_and_processing_time(self) -> None:
        status, payload = self.post_text("Ping dana.reyes@example.org today.")

        self.assertEqual(status, 200)
        self.assertEqual(
            [span["label"] for span in payload["spans"]], ["private_email"]
        )
        self.assertIsInstance(payload["processing_ms"], int)
        self.assertGreaterEqual(payload["processing_ms"], 0)


class StubServer:
    """Stands in for ThreadingHTTPServer so main() never binds a socket."""

    def __init__(self, interrupt_on_serve: bool = False) -> None:
        self.interrupt_on_serve = interrupt_on_serve
        self.bound_address: tuple[str, int] | None = None
        self.handler: type | None = None
        self.serve_forever_called = False
        self.shutdown_called = False
        self.shutdown_thread: threading.Thread | None = None
        self.closed = False

    def build(self, address: tuple[str, int], handler: type) -> StubServer:
        self.bound_address = address
        self.handler = handler
        return self

    def serve_forever(self) -> None:
        self.serve_forever_called = True
        if self.interrupt_on_serve:
            raise KeyboardInterrupt

    def shutdown(self) -> None:
        self.shutdown_thread = threading.current_thread()
        self.shutdown_called = True

    def server_close(self) -> None:
        self.closed = True


class MainWiringTests(TestCase):
    def test_termination_signals_are_wired_to_server_shutdown(self) -> None:
        stub = StubServer()
        with (
            patch.object(sys, "argv", ["pii-server.py", "--mode", "rules"]),
            patch.object(PII_SERVER.signal, "signal") as install,
            patch.object(PII_SERVER, "ThreadingHTTPServer", stub.build),
        ):
            PII_SERVER.main()

        handlers = {call.args[0]: call.args[1] for call in install.call_args_list}
        self.assertEqual(set(handlers), {signal.SIGINT, signal.SIGTERM})
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        self.assertTrue(wait_until(lambda: stub.shutdown_called))
        self.assertIs(stub.handler, PII_SERVER.Handler)
        self.assertEqual(stub.bound_address, ("127.0.0.1", 9123))
        self.assertEqual(PII_SERVER.Handler.mode, "rules")

    def test_shutdown_runs_off_the_serving_thread(self) -> None:
        stub = StubServer()
        with (
            patch.object(sys, "argv", ["pii-server.py", "--mode", "rules"]),
            patch.object(PII_SERVER.signal, "signal") as install,
            patch.object(PII_SERVER, "ThreadingHTTPServer", stub.build),
        ):
            PII_SERVER.main()

        handlers = {call.args[0]: call.args[1] for call in install.call_args_list}
        handlers[signal.SIGTERM](signal.SIGTERM, None)

        self.assertTrue(wait_until(lambda: stub.shutdown_called))
        shutdown_thread = cast(threading.Thread, stub.shutdown_thread)
        self.assertIsNot(shutdown_thread, threading.current_thread())
        self.assertTrue(shutdown_thread.daemon)

    def test_keyboard_interrupt_during_serve_closes_the_server(self) -> None:
        stub = StubServer(interrupt_on_serve=True)
        with (
            patch.object(sys, "argv", ["pii-server.py", "--mode", "rules"]),
            patch.object(PII_SERVER.signal, "signal"),
            patch.object(PII_SERVER, "ThreadingHTTPServer", stub.build),
        ):
            PII_SERVER.main()

        self.assertTrue(stub.serve_forever_called)
        self.assertTrue(stub.closed)


class ScriptEntryPointTests(TestCase):
    """Run the module as a script, the way the README starts it."""

    def start_server(self, port: int) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [
                sys.executable,
                str(MODULE_PATH),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--mode",
                "rules",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def stop_server(self, process: subprocess.Popen[str]) -> tuple[str, str]:
        """Leave nothing behind, and hand back what the process wrote."""
        if process.poll() is None:
            process.kill()
        return process.communicate(timeout=20)

    def test_script_serves_rules_mode_over_http(self) -> None:
        port = free_port()
        process = self.start_server(port)
        try:
            self.assertEqual(
                wait_for_health(port),
                (
                    200,
                    {"status": "ok", "mode": "rules", "device": "cpu", "busy": False},
                ),
            )
            status, payload = request_json(
                port,
                "POST",
                "/",
                json.dumps({"text": "Ping dana.reyes@example.org today."}).encode(
                    "utf-8"
                ),
            )
            self.assertEqual(status, 200)
            self.assertEqual(
                [span["label"] for span in payload["spans"]], ["private_email"]
            )
        finally:
            _, stderr = self.stop_server(process)

        self.assertIn("[rules] loading model...", stderr)
        self.assertIn(f"[rules] ready on http://127.0.0.1:{port}", stderr)
        self.assertRegex(stderr, r"\[rules\] rules engine: (native|python \(.+\))\n")

    def test_sigterm_stops_the_server_process(self) -> None:
        """The regression test for the signal deadlock.

        The handler used to call shutdown() on the thread that was inside
        serve_forever(), so the process ignored SIGTERM, held the port, and
        answered nothing. Only a live process shows that.
        """
        port = free_port()
        process = self.start_server(port)
        try:
            wait_for_health(port)
            process.send_signal(signal.SIGTERM)

            self.assertEqual(process.wait(timeout=10), 0)
            with self.assertRaises(OSError):
                request_json(port, "GET", "/health")
        finally:
            self.stop_server(process)
