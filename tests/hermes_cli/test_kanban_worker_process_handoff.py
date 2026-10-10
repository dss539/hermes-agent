"""Real dispatcher + CLI worker + SQLite, against a scripted loopback provider."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def test_spawned_worker_recovers_rejected_completion(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch

    requests = []
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "rejected", "type": "function",
            "function": {"name": "kanban_complete", "arguments": "{}"}}]},
        {"role": "assistant", "content": "Work is complete."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "accepted", "type": "function",
            "function": {"name": "kanban_complete", "arguments": json.dumps({"summary": "Verified local process contract"})}}]},
        {"role": "assistant", "content": "Board handoff committed."},
    ]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            if self.path != "/v1/chat/completions" or len(requests) > len(messages):
                self.send_error(400, "Unexpected fixture request")
                return
            message = messages[len(requests) - 1]
            finish = "tool_calls" if message.get("tool_calls") else "stop"
            base = {"id": "fixture", "created": 1, "model": "scripted-local"}
            if body.get("stream"):
                delta = dict(message)
                if delta.get("tool_calls"):
                    delta["tool_calls"] = [dict(call, index=i) for i, call in enumerate(delta["tool_calls"])]
                chunks = [
                    dict(base, object="chat.completion.chunk", choices=[{"index": 0, "delta": delta, "finish_reason": None}]),
                    dict(base, object="chat.completion.chunk", choices=[{"index": 0, "delta": {}, "finish_reason": finish}]),
                ]
                mime = "text/event-stream"
                data = ("".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
            else:
                mime = "application/json"
                data = json.dumps(dict(base, object="chat.completion", choices=[{
                    "index": 0, "message": message, "finish_reason": finish}])).encode()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "isolated-state"
    root.mkdir()
    observer = tmp_path / "observer"
    observer.mkdir()
    # External requests are forbidden; the child may contact only this local fixture.
    observer.joinpath("sitecustomize.py").write_text("""
import sys

def local_only(event, args):
    if event == 'socket.getaddrinfo' and args[0] not in ('127.0.0.1', 'localhost', '::1'):
        raise RuntimeError('process fixture forbids external DNS')
    if event == 'socket.connect' and isinstance(args[1], tuple) and args[1][0] not in ('127.0.0.1', '::1'):
        raise RuntimeError('process fixture forbids external sockets')
sys.addaudithook(local_only)
""")
    root.joinpath("config.yaml").write_text(f"""
model:
  provider: custom
  default: scripted-local
  base_url: http://127.0.0.1:{server.server_port}/v1
  api_mode: chat_completions
  context_length: 131072
nous:
  guest: false
platform_toolsets:
  cli: [kanban]
toolsets: [kanban]
compression:
  enabled: false
""")
    monkeypatch.setattr(os, "environ", {
        "HOME": str(home), "USERPROFILE": str(home), "HERMES_HOME": str(root),
        "HERMES_TEST_ISOLATION": str(root), "HERMES_GUEST_ONBOARDING": "0",
        "HERMES_DISABLE_LAZY_INSTALLS": "1", "HERMES_DISABLE_PLUGINS": "1",
        "AWS_EC2_METADATA_DISABLED": "true",
        "HERMES_TUI": "0", "NO_PROXY": "127.0.0.1,localhost",
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
        "PYTHONPATH": str(observer),
        **{key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ},
    })
    monkeypatch.setattr(Path, "home", lambda: home)
    workspace = tmp_path / "task"
    workspace.mkdir()
    marker = workspace / "inspect-imported"
    workspace.joinpath("inspect.py").write_text(f"open({str(marker)!r}, 'w').write('unexpected import')\n")
    children = []
    real_popen = subprocess.Popen

    def retain_child(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", retain_child)
    try:
        kb.init_db()
        with kbc.connect_closing() as conn:
            task_id = kb.create_task(conn, title="Local scripted handoff", assignee="default",
                                     workspace_kind="dir", workspace_path=str(workspace))
            result = dispatch.dispatch_once(conn, max_spawn=1, max_in_progress=1, reconcile_orphans=False)
            assert [row[0] for row in result.spawned] == [task_id], result
            run_id = kb.get_task(conn, task_id).current_run_id
        assert len(children) == 1
        child = children[0]
        child.wait(timeout=60)
        logs = "\n".join(p.read_text() for p in root.rglob(f"{task_id}.log"))
        assert not marker.exists(), logs
        assert child.returncode == 0, logs
        assert "Session store unavailable" not in logs, logs
        with sqlite3.connect(root / "state.db") as session_db:
            assert session_db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert len(requests) == 4, logs
        assert any(m.get("role") == "tool" and "error" in m.get("content", "") for m in requests[1]["messages"])
        assert any("No terminal board handoff has committed" in str(m.get("content")) for m in requests[2]["messages"])
        with kbc.connect_closing() as conn:
            task = kb.get_task(conn, task_id)
            assert task.status == "done"
            assert task.consecutive_failures == 0
            assert kb.goal_run_status(conn, task_id, run_id) == "done"
            run = conn.execute("SELECT outcome FROM task_runs WHERE id=?", (run_id,)).fetchone()
            assert run["outcome"] == "completed"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
