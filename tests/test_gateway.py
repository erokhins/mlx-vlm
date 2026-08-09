import asyncio
import json
import os
import signal
import sys
import threading
import time

import httpx
from fastapi.testclient import TestClient

import mlx_vlm_gateway.supervisor as supervisor_module
from mlx_vlm_gateway.app import GatewaySettings, create_app
from mlx_vlm_gateway.memory_monitor import MemorySample
from mlx_vlm_gateway.supervisor import GATEWAY_PID_ENV


class FakeProcess:
    _next_pid = 41000

    def __init__(self):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = None
        self._done = asyncio.Event()

    def terminate(self):
        self.returncode = 0
        self._done.set()

    def kill(self):
        self.returncode = -9
        self._done.set()

    async def wait(self):
        await self._done.wait()
        return self.returncode


def _gateway(
    monkeypatch,
    handler,
    *,
    shutdown_callback=None,
    memory_sampler=None,
    **settings_overrides,
):
    processes = []

    async def fake_create_subprocess(*_args, **_kwargs):
        process = FakeProcess()
        process.spawn_kwargs = _kwargs
        processes.append(process)
        return process

    def fake_killpg(pid, sig):
        process = next(process for process in processes if process.pid == pid)
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()

    monkeypatch.setattr(
        supervisor_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess,
    )
    monkeypatch.setattr(supervisor_module.os, "killpg", fake_killpg, raising=False)

    transport = httpx.MockTransport(handler)

    def client_factory(timeout):
        return httpx.AsyncClient(transport=transport, timeout=timeout)

    setting_values = dict(
        worker_command=(sys.executable, "-c", "pass"),
        startup_timeout_s=1.0,
        request_timeout_s=0.5,
        startup_probe_interval_s=0.01,
        probe_interval_s=0.02,
        probe_timeout_s=0.1,
        restart_delay_s=0.01,
        shutdown_timeout_s=0.1,
        idle_check_interval_s=0.01,
    )
    setting_values.update(settings_overrides)
    settings = GatewaySettings(**setting_values)
    return (
        create_app(
            settings,
            client_factory=client_factory,
            shutdown_callback=shutdown_callback,
            memory_sampler=memory_sampler,
        ),
        processes,
    )


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_gateway_records_memory_for_the_current_worker(monkeypatch):
    sampled_pids = []

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        raise AssertionError(request.url.path)

    def memory_sampler(pid):
        sampled_pids.append(pid)
        return MemorySample(
            timestamp=time.time(),
            pressure="normal",
            available_bytes=100,
            worker_bytes=50,
            worker_pid=pid,
        )

    app, processes = _gateway(
        monkeypatch, handler, memory_sampler=memory_sampler
    )
    with TestClient(app):
        _wait_until(lambda: len(app.state.memory_samples) > 0)
        sample = app.state.memory_samples.snapshot()[-1]

    assert sample.worker_pid == processes[0].pid
    assert sampled_pids[0] == processes[0].pid


def test_worker_output_goes_to_the_configured_log(monkeypatch, tmp_path):
    log = tmp_path / "junie-mlx-vlm.log"

    def handler(request):
        return httpx.Response(200, json={"status": "ready"})

    app, processes = _gateway(monkeypatch, handler, worker_log_path=str(log))
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)
        spawned = processes[0].spawn_kwargs

    # One file for both streams, opened for append so restarts within a run
    # add to it instead of truncating each other.
    assert spawned["stdout"] is spawned["stderr"]
    assert spawned["stdout"].name == str(log)
    assert spawned["stdout"].mode == "a"


def test_worker_inherits_our_output_when_no_log_is_configured(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"status": "ready"})

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)
        spawned = processes[0].spawn_kwargs

    assert "stdout" not in spawned and "stderr" not in spawned


def test_gateway_forwards_batch_requests_and_controls_worker(monkeypatch):
    captured = []

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(
                200,
                json={"status": "ready", "loaded_model": "demo"},
            )
        if request.url.path == "/v1/chat/completions":
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "timings": {"generation_tps": 42.0},
                },
            )
        if request.url.path == "/metrics":
            return httpx.Response(200, json={"requests": {"completed": 1}})
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={"status": "healthy", "loaded_model": "demo"},
            )
        if request.url.path == "/cache/stats":
            return httpx.Response(200, json={"enabled": True})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "demo", "object": "model", "created": 1}],
                },
            )
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)
        assert processes[0].spawn_kwargs["env"][GATEWAY_PID_ENV] == str(os.getpid())
        assert client.post("/start_worker").status_code == 404
        assert client.post("/stop_worker").status_code == 404
        response = client.post(
            "/v1/chat/completions",
            json={"model": "any-model", "messages": [], "stream": True},
        )
        assert response.status_code == 200
        assert response.json()["timings"]["generation_tps"] == 42.0
        assert captured == [{"model": "any-model", "messages": [], "stream": False}]

        metrics = client.get("/metrics").json()
        assert metrics["requests"]["completed"] == 1
        assert metrics["gateway"]["requests_completed"] == 1
        assert client.get("/v1/metrics").status_code == 200
        assert client.get("/health").json() == {
            "status": "healthy",
            "loaded_model": "demo",
        }
        assert client.get("/cache/stats").json() == {"enabled": True}
        assert client.get("/v1/cache/stats").json() == {"enabled": True}
        assert client.post("/cache/reset").status_code == 404
        assert client.post("/v1/cache/reset").status_code == 404
        assert client.get("/v1/models").json()["data"][0]["id"] == "demo"

        assert client.post("/unload").json() == {
            "status": "success",
            "message": "Model unloaded successfully",
            "unloaded": {
                "model_name": "demo",
                "adapter_name": None,
                "models": {},
            },
        }
        assert client.get("/ready").status_code == 503
        process_count = len(processes)
        time.sleep(0.06)
        assert len(processes) == process_count
        assert client.post("/v1/unload").json() == {
            "status": "no_model_loaded",
            "message": "No model is currently loaded",
        }
        assert client.get("/v1/models").json()["data"][0]["id"] == "demo"

        assert client.post("/v1/chat/completions", json={}).status_code == 200
        assert len(processes) == process_count + 1


def test_junie_status_and_settings_endpoints(monkeypatch, tmp_path):
    config_path = tmp_path / "server-config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "demo-model",
                "draft_model": "demo-draft",
                "max_context_length": 12345,
                "kv_quantization": True,
                "auto_unload_time": 600,
            }
        )
    )

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(
                200,
                json={"status": "ready", "loaded_model": "demo-model"},
            )
        raise AssertionError(request.url.path)

    app, _ = _gateway(monkeypatch, handler, config_path=str(config_path))
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "ready")
        expected_settings = {
            "model_name": "demo-model",
            "max_context_length": 12345,
            "kv_quantization": True,
            "auto_unload_time": 600,
        }
        assert client.get("/current_settings").json() == expected_settings
        assert client.get("/v1/current_settings").json() == expected_settings

        status = client.get("/status").json()
        alias_status = client.get("/v1/status").json()
        assert alias_status["uptime_s"] >= status["uptime_s"]
        alias_status["uptime_s"] = status["uptime_s"]
        assert alias_status == status
        assert status["phase"] == "ready"
        assert status["model"] == {
            "loaded": True,
            "id": "demo-model",
            "draft_model": "demo-draft",
            "context_limit": 12345,
        }
        assert status["memory"] == {}
        assert status["inference"] == {
            "in_progress": False,
            "in_flight": 0,
            "queue_depth": 0,
            "requests": [],
        }


def test_status_reports_worker_memory_from_ready_probe(monkeypatch, tmp_path):
    config_path = tmp_path / "server-config.json"
    config_path.write_text(json.dumps({"model_name": "demo-model"}))

    memory = {"total_gb": 19.06, "peak_gb": 21.49, "kv_cache_gb": 1.9}

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(
                200,
                json={
                    "status": "ready",
                    "loaded_model": "demo-model",
                    "memory": memory,
                },
            )
        raise AssertionError(request.url.path)

    app, _ = _gateway(monkeypatch, handler, config_path=str(config_path))
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "ready")
        assert client.get("/status").json()["memory"] == memory


def test_apply_auto_unload_time_without_restarting_worker(monkeypatch, tmp_path):
    config_path = tmp_path / "server-config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "demo",
                "auto_unload_time": None,
                "internal": 7,
            }
        )
    )

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler, config_path=str(config_path))
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "ready")
        app.state.supervisor.active_requests = 1
        response = client.post("/apply_settings", json={"auto_unload_time": 600})
        app.state.supervisor.active_requests = 0

        assert response.status_code == 200
        assert response.json() == {
            "status": "applied",
            "changes": ["auto_unload_time"],
            "settings": {
                "model_name": "demo",
                "max_context_length": None,
                "kv_quantization": True,
                "auto_unload_time": 600,
            },
        }
        assert len(processes) == 1
        assert json.loads(config_path.read_text())["internal"] == 7


def test_invalid_auto_unload_uses_default_and_task_keeps_running(monkeypatch, tmp_path):
    config_path = tmp_path / "server-config.json"
    config_path.write_text(
        json.dumps({"model_name": "demo", "auto_unload_time": "hello"})
    )

    captured = []

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(
                200,
                json={"status": "ready", "loaded_model": "demo"},
            )
        if request.url.path == "/v1/chat/completions":
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler, config_path=str(config_path))
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "ready")
        assert json.loads(config_path.read_text())["auto_unload_time"] == 600
        app.state.supervisor.last_activity_at -= 601
        _wait_until(lambda: processes[0].returncode is not None)

        status = client.get("/status").json()
        assert status["phase"] == "ready"
        assert status["model"]["loaded"] is False
        assert app.state.supervisor.desired_running is False

        response = client.post(
            "/v1/chat/completions",
            json={"model": "demo", "messages": [], "stream": True},
        )
        assert response.status_code == 200
        assert len(processes) == 2
        assert captured == [{"model": "demo", "messages": [], "stream": False}]


def test_apply_restart_setting_rejects_busy_request_without_force(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "server-config.json"
    config_path.write_text(json.dumps({"model_name": "demo"}))

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler, config_path=str(config_path))
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "ready")
        app.state.supervisor.active_requests = 1
        response = client.post(
            "/v1/apply_settings",
            json={"max_context_length": 150000},
        )

        assert response.status_code == 409
        assert response.json() == {
            "detail": (
                '1 inference request(s) in flight; pass "force": true '
                "to restart model serving anyway."
            )
        }
        assert len(processes) == 1
        assert json.loads(config_path.read_text())["max_context_length"] is None


def test_config_save_failure_keeps_worker_running(monkeypatch, tmp_path):
    config_path = tmp_path / "server-config.json"
    config_path.write_text(json.dumps({"model_name": "demo", "kv_quantization": False}))

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler, config_path=str(config_path))
    with TestClient(app, raise_server_exceptions=False) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "ready")

        def fail_save(_updates):
            raise OSError("disk full")

        monkeypatch.setattr(app.state.settings_store, "save", fail_save)
        response = client.post(
            "/apply_settings",
            json={"kv_quantization": True},
        )

        assert response.status_code == 500
        assert response.json() == {
            "detail": "Failed to save settings; worker state was not changed."
        }
        assert len(processes) == 1
        assert processes[0].returncode is None
        assert json.loads(config_path.read_text())["kv_quantization"] is False
        assert [item.name for item in tmp_path.iterdir()] == ["server-config.json"]


def test_applying_current_settings_is_a_noop(monkeypatch, tmp_path):
    model = "mlx-community/Qwen3.6-27B-4bit"
    config_path = tmp_path / "server-config.json"
    current = {
        "model_name": model,
        "max_context_length": 12345,
        "kv_quantization": False,
        "auto_unload_time": 600,
    }
    config_path.write_text(json.dumps(current))

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler, config_path=str(config_path))
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "ready")

        original_process = processes[0]
        response = client.post("/apply_settings", json=current)

        assert response.status_code == 200
        assert response.json() == {
            "status": "applied",
            "changes": [],
            "settings": current,
        }
        assert len(processes) == 1
        assert processes[0] is original_process


def test_force_apply_restarts_worker_without_stale_request_restart(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "server-config.json"
    config_path.write_text(json.dumps({"model_name": "demo", "kv_quantization": False}))
    inference_started = threading.Event()
    processes = None

    async def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            inference_started.set()
            while len(processes) < 2:
                await asyncio.sleep(0.005)
            raise httpx.ConnectError("old worker stopped", request=request)
        raise AssertionError(request.url.path)

    app, processes = _gateway(
        monkeypatch,
        handler,
        config_path=str(config_path),
    )
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "ready")
        result = {}

        def send_request():
            result["response"] = client.post("/v1/chat/completions", json={})

        request_thread = threading.Thread(target=send_request)
        request_thread.start()
        assert inference_started.wait(timeout=1.0)

        response = client.post(
            "/apply_settings",
            json={"kv_quantization": True, "force": True},
        )
        request_thread.join(timeout=1.0)

        assert response.status_code == 200
        assert response.json() == {
            "status": "applying",
            "model": "demo",
            "changes": ["kv_quantization"],
            "message": (
                "Model serving is restarting; poll GET /status until phase is 'ready'."
            ),
        }
        assert not request_thread.is_alive()
        assert result["response"].status_code == 503
        time.sleep(0.06)
        assert len(processes) == 2
        assert json.loads(config_path.read_text())["kv_quantization"] is True


def test_shutdown_stops_worker_and_gateway_accepts_v1_alias(monkeypatch):
    shutdown_called = threading.Event()

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(
        monkeypatch,
        handler,
        shutdown_callback=shutdown_called.set,
    )
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "ready")

        response = client.post("/v1/shutdown")

        assert response.status_code == 200
        assert response.json() == {"status": "shutting_down"}
        assert processes[0].returncode == 0
        assert client.get("/status").json()["phase"] == "stopping"
        assert client.post("/v1/chat/completions", json={}).status_code == 503
        assert shutdown_called.wait(timeout=1.0)


def test_second_consecutive_500_restarts_worker(monkeypatch):
    inference_calls = 0

    def handler(request):
        nonlocal inference_calls
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            inference_calls += 1
            return httpx.Response(500, json={"detail": "generation failed"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)
        assert client.post("/v1/chat/completions", json={}).status_code == 500
        assert len(processes) == 1
        second = client.post("/v1/chat/completions", json={})
        assert second.status_code == 503
        assert inference_calls == 2
        _wait_until(lambda: len(processes) == 2)


def test_worker_508_returns_503_and_restarts_worker(monkeypatch):
    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                508, json={"detail": "corrupted generation"}
            )
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)
        assert len(processes) == 1

        response = client.post("/v1/chat/completions", json={})

        # A single 508 is enough: 503 to the client, fresh worker process.
        assert response.status_code == 503
        assert "restarting" in response.json()["detail"]
        _wait_until(lambda: len(processes) == 2)


def test_confirmed_worker_oom_returns_error_and_restarts_worker(monkeypatch):
    error_payload = {
        "error": {
            "message": "The inference worker ran out of memory and is restarting.",
            "type": "server_error",
            "code": "out_of_memory",
        }
    }

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(503, json=error_payload)
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)
        assert len(processes) == 1

        response = client.post("/v1/chat/completions", json={})

        assert response.status_code == 503
        assert response.json() == error_payload
        _wait_until(lambda: len(processes) == 2)


def test_worker_503_without_oom_code_does_not_restart_worker(monkeypatch):
    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(503, json={"detail": "temporarily unavailable"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)

        response = client.post("/v1/chat/completions", json={})

        assert response.status_code == 503
        assert len(processes) == 1


def test_client_disconnect_aborts_worker_request(monkeypatch):
    # TestClient cannot hang up mid-request, so this drives the ASGI app
    # directly: body first, then http.disconnect while the worker "runs".
    worker_saw_cancel = []

    async def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                worker_saw_cancel.append(True)
                raise
        raise AssertionError(request.url.path)

    app, _ = _gateway(monkeypatch, handler)

    async def run():
        async with app.router.lifespan_context(app):
            sup = app.state.supervisor
            async def until_ready():
                while sup.state != "ready":
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(until_ready(), timeout=2.0)

            messages = [
                {
                    "type": "http.request",
                    "body": json.dumps({"messages": []}).encode(),
                    "more_body": False,
                },
                {"type": "http.disconnect"},
            ]

            async def receive():
                return messages.pop(0) if messages else {"type": "http.disconnect"}

            sent = []

            async def send(message):
                sent.append(message)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/chat/completions",
                "raw_path": b"/v1/chat/completions",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 4321),
                "server": ("127.0.0.1", 80),
            }
            await asyncio.wait_for(app(scope, receive, send), timeout=2.0)
            return sent

    sent = asyncio.run(run())
    sup = app.state.supervisor

    assert worker_saw_cancel, "worker request was not aborted"
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 499
    assert sup.active_requests == 0
    assert sup.requests_cancelled == 1
    assert sup.requests_failed == 0


def test_old_worker_responses_do_not_affect_new_worker(monkeypatch):
    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(
        monkeypatch,
        handler,
        restart_delay_s=0.05,
    )
    with TestClient(app) as client:
        supervisor = app.state.supervisor

        def record_500(generation):
            return client.portal.call(
                lambda: supervisor.record_worker_response(
                    500, generation=generation
                )
            )

        _wait_until(lambda: supervisor.state == "ready")
        old_generation = supervisor.generation

        assert record_500(old_generation) is False
        assert supervisor.consecutive_500 == 1

        client.portal.call(
            lambda: supervisor.schedule_restart(
                "test restart", generation=old_generation
            )
        )
        _wait_until(lambda: len(processes) == 2 and supervisor.state == "ready")
        new_generation = supervisor.generation
        assert new_generation != old_generation
        assert supervisor.consecutive_500 == 0

        assert record_500(old_generation) is False
        assert supervisor.consecutive_500 == 0

        assert record_500(new_generation) is False
        assert supervisor.consecutive_500 == 1

        assert record_500(new_generation) is True
        assert supervisor.state == "restarting"


def test_repeated_start_failures_stop_restart_loop(monkeypatch):
    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(503, json={"status": "loading"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(
        monkeypatch,
        handler,
        startup_timeout_s=0.03,
        max_start_failures=3,
    )
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "error")
        status = client.get("/status").json()
        assert status["phase_detail"] == "worker startup timeout"
        assert len(processes) == 3

        process_count = len(processes)
        time.sleep(0.08)
        assert len(processes) == process_count

        response = client.post(
            "/apply_settings", json={"max_context_length": 12345}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "applying"
        _wait_until(lambda: len(processes) == process_count + 1)


def test_process_spawn_failures_leave_gateway_available(monkeypatch):
    def handler(request):
        raise AssertionError(request.url.path)

    app, _ = _gateway(
        monkeypatch,
        handler,
        max_start_failures=3,
    )
    spawn_attempts = 0

    async def fail_to_spawn(*_args, **_kwargs):
        nonlocal spawn_attempts
        spawn_attempts += 1
        raise FileNotFoundError("worker executable missing")

    monkeypatch.setattr(
        supervisor_module.asyncio,
        "create_subprocess_exec",
        fail_to_spawn,
    )

    with TestClient(app) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "error")

        status = client.get("/status").json()
        assert status["phase_detail"] == (
            "Failed to start inference worker: worker executable missing"
        )
        assert spawn_attempts == 3

        for _ in range(2):
            response = client.post("/v1/chat/completions", json={})
            assert response.status_code == 503
            assert response.json()["detail"] == (
                "Inference worker is recovering; please retry shortly."
            )

        time.sleep(0.08)
        assert spawn_attempts == 3


def test_request_retries_worker_after_startup_cooldown(monkeypatch):
    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": []})
        raise AssertionError(request.url.path)

    app, processes = _gateway(
        monkeypatch,
        handler,
        max_start_failures=3,
        startup_retry_cooldown_s=0.03,
    )
    spawn_attempts = 0

    async def recover_on_fourth_spawn(*_args, **_kwargs):
        nonlocal spawn_attempts
        spawn_attempts += 1
        if spawn_attempts <= 3:
            raise OSError("temporary startup failure")
        process = FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(
        supervisor_module.asyncio,
        "create_subprocess_exec",
        recover_on_fourth_spawn,
    )

    with TestClient(app) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "error")
        assert spawn_attempts == 3

        assert client.post("/v1/chat/completions", json={}).status_code == 503
        time.sleep(0.05)
        assert spawn_attempts == 3

        response = client.post("/v1/chat/completions", json={})

        assert response.status_code == 200
        assert spawn_attempts == 4
        assert client.get("/status").json()["phase"] == "ready"


def test_422_between_500_responses_resets_restart_counter(monkeypatch):
    statuses = iter((500, 422, 500))

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(next(statuses), json={"detail": "test"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)
        assert client.post("/v1/chat/completions", json={}).status_code == 500
        assert client.post("/v1/chat/completions", json={}).status_code == 422
        assert client.post("/v1/chat/completions", json={}).status_code == 500
        time.sleep(0.06)
        assert len(processes) == 1


def test_worker_connection_failure_returns_503_and_restarts(monkeypatch):
    fail_inference = True

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions" and fail_inference:
            raise httpx.ConnectError("worker exited", request=request)
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)
        response = client.post("/v1/chat/completions", json={})
        assert response.status_code == 503
        assert response.json()["detail"] == "Inference worker restarted; please retry"
        _wait_until(lambda: len(processes) == 2)


def test_worker_connection_failure_with_fresh_oom_log_returns_oom(
    monkeypatch, tmp_path
):
    log = tmp_path / "worker.log"

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            with log.open("a") as stream:
                stream.write(
                    "[METAL] Command buffer execution failed: Insufficient Memory\n"
                )
            raise httpx.ConnectError("worker exited", request=request)
        raise AssertionError(request.url.path)

    app, processes = _gateway(
        monkeypatch, handler, worker_log_path=str(log)
    )
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)

        response = client.post("/v1/chat/completions", json={})

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "out_of_memory"
        _wait_until(lambda: len(processes) == 2)


def test_worker_connection_failure_ignores_stale_oom_log(monkeypatch, tmp_path):
    log = tmp_path / "worker.log"
    log.write_text("[METAL] Command buffer execution failed: Insufficient Memory\n")

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            raise httpx.ConnectError("worker exited", request=request)
        raise AssertionError(request.url.path)

    app, processes = _gateway(
        monkeypatch, handler, worker_log_path=str(log)
    )
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)

        response = client.post("/v1/chat/completions", json={})

        assert response.status_code == 503
        assert response.json()["detail"] == "Inference worker restarted; please retry"
        _wait_until(lambda: len(processes) == 2)


def test_hard_timeout_returns_504_and_restarts_worker(monkeypatch):
    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            raise httpx.ReadTimeout("generation did not stop", request=request)
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)
        response = client.post("/v1/chat/completions", json={})
        assert response.status_code == 504
        assert response.json()["detail"] == "Inference worker did not stop in time"
        _wait_until(lambda: len(processes) == 2)


def test_unload_interrupts_active_request_without_restart(monkeypatch):
    inference_started = threading.Event()
    processes = None

    async def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            inference_started.set()
            while processes[0].returncode is None:
                await asyncio.sleep(0.005)
            raise httpx.ConnectError("worker stopped", request=request)
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/ready").status_code == 200)
        result = {}

        def send_request():
            result["response"] = client.post("/v1/chat/completions", json={})

        request_thread = threading.Thread(target=send_request)
        request_thread.start()
        assert inference_started.wait(timeout=1.0)

        assert client.post("/v1/unload").status_code == 200
        request_thread.join(timeout=1.0)
        assert not request_thread.is_alive()
        assert result["response"].status_code == 503

        process_count = len(processes)
        time.sleep(0.06)
        assert len(processes) == process_count
        assert client.get("/ready").status_code == 503


def test_unload_interrupts_request_waiting_for_worker_startup(monkeypatch):
    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(503, json={"status": "loading"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(
        monkeypatch,
        handler,
        startup_timeout_s=1.0,
    )
    with TestClient(app) as client:
        _wait_until(
            lambda: client.get("/status").json()["phase"] == "loading_model"
        )
        result = {}

        def send_request():
            result["response"] = client.post("/v1/chat/completions", json={})

        request_thread = threading.Thread(target=send_request)
        request_thread.start()
        _wait_until(lambda: app.state.supervisor.active_requests == 1)

        assert client.post("/unload").status_code == 200
        request_thread.join(timeout=1.0)

        assert not request_thread.is_alive()
        assert result["response"].status_code == 503
        assert processes[0].returncode == 0
        assert app.state.supervisor.desired_running is False

        time.sleep(0.06)
        assert len(processes) == 1


def test_unload_during_restart_delay_prevents_worker_respawn(monkeypatch):
    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(
        monkeypatch,
        handler,
        restart_delay_s=0.1,
    )
    with TestClient(app) as client:
        _wait_until(lambda: client.get("/status").json()["phase"] == "ready")

        client.portal.call(app.state.supervisor.schedule_restart, "test restart")
        _wait_until(
            lambda: app.state.supervisor.state == "restarting"
            and app.state.supervisor.process is None
        )

        response = client.post("/unload")

        assert response.status_code == 200
        assert response.json()["status"] == "no_model_loaded"
        assert app.state.supervisor.desired_running is False
        time.sleep(0.15)
        assert len(processes) == 1
