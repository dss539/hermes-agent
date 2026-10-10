"""Real worker boot must not import Python files from its task directory."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


# Runs BEFORE Hermes imports, without importing inspect ourselves (which would
# conceal the regression). Observes the actual CLI child at exit, and forbids
# network access even if first-run behaviour later changes.
_OBSERVER = r'''
import atexit
import os
import sys

network_attempts = []
def no_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        network_attempts.append(event)
        raise RuntimeError("worker import regression must remain offline")
sys.addaudithook(no_network)

def report():
    import json
    inspect = sys.modules.get("inspect")
    bootstrap = sys.modules.get("hermes_cli.free_tier_bootstrap")
    data = {
        "inspect": getattr(inspect, "__file__", None),
        "cli_file": getattr(sys.modules.get("hermes_cli.main") or sys.modules.get("__main__"), "__file__", None),
        "cwd": os.getcwd(),
        "home": os.environ.get("HERMES_HOME"),
        "terminal_cwd": os.environ.get("TERMINAL_CWD"),
        "workspace": os.environ.get("HERMES_KANBAN_WORKSPACE"),
        "board": os.environ.get("HERMES_KANBAN_BOARD"),
        "task": os.environ.get("HERMES_KANBAN_TASK"),
        "sys_path": sys.path,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "bootstrapped": bool(bootstrap and bootstrap.current_record()),
        "network_attempts": network_attempts,
    }
    with open(os.environ["KANBAN_IMPORT_RECEIPT"], "w") as f:
        json.dump(data, f)
atexit.register(report)
'''


@pytest.mark.parametrize("path_entry, custom_bin", [
    (None, False), ("", False), (".", False), ("absolute", False),
    ("alias", False), ("absolute", True),
])
def test_worker_boot_ignores_task_local_inspect(tmp_path, monkeypatch, path_entry, custom_bin):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as dispatch

    # No real home, credentials, host auth CLI or profile participates. This is
    # a temporary directory fixture, NOT a profile-management command.
    home = tmp_path / "home"
    root = home / ".hermes"
    profile = root / "profiles" / "reviewer"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("nous:\n  guest: false\n", encoding="utf-8")
    observer = tmp_path / "observer"
    observer.mkdir()
    observer.joinpath("sitecustomize.py").write_text(_OBSERVER, encoding="utf-8")
    workspace = tmp_path / "task"
    workspace.mkdir()
    marker = workspace / "inspect-imported"
    workspace.joinpath("inspect.py").write_text(
        f"open({str(marker)!r}, 'w').write('task inspect imported')\n",
        encoding="utf-8",
    )
    if path_entry == "absolute":
        path_entry = str(workspace)
    elif path_entry == "alias":
        alias = tmp_path / "task-alias"
        try:
            alias.symlink_to(workspace, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlinks unavailable")
        path_entry = str(alias)
    pythonpath = str(observer)
    if path_entry is not None:
        pythonpath += os.pathsep + path_entry
    receipt = tmp_path / "receipt.json"
    # Clear inherited API keys as well as Python launch switches (e.g. an
    # inherited PYTHONSAFEPATH could make the broken base spuriously pass).
    monkeypatch.setattr(os, "environ", {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "HERMES_HOME": str(root),
        "HERMES_TEST_ISOLATION": str(root),
        "HERMES_GUEST_ONBOARDING": "0",
        "HERMES_DISABLE_LAZY_INSTALLS": "1",
        # Bedrock is installed in the current runtime: never probe EC2 credentials.
        "AWS_EC2_METADATA_DISABLED": "true",
        "HERMES_TUI": "0",
        "PATH": str(Path(sys.executable).parent),
        "PYTHONPATH": pythonpath,
        "KANBAN_IMPORT_RECEIPT": str(receipt),
        **{key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ},
    })
    if custom_bin:
        if os.name == "nt":
            pytest.skip("POSIX executable shim")
        shim = observer / "custom hermes"
        shim.write_text(
            f"#!{sys.executable}\nimport sys\n"
            f"sys.path.insert(0, {str(Path(dispatch.__file__).resolve().parents[1])!r})\n"
            "from hermes_cli.main import main\nmain()\n",
            encoding="utf-8",
        )
        shim.chmod(0o700)
        monkeypatch.setenv("HERMES_BIN", str(shim))
    task = kb.Task(
        id="t_import_path", title="offline startup probe", body=None,
        assignee="reviewer", status="running", priority=0, created_by="test",
        created_at=1, started_at=None, completed_at=None, workspace_kind="dir",
        workspace_path=str(workspace), claim_lock="lock", claim_expires=None,
        tenant=None, current_run_id=1,
    )
    real_popen = subprocess.Popen
    children = []

    def retain_child(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    # Retain a handle only: no argv, environment, imports or cwd are replaced.
    monkeypatch.setattr(subprocess, "Popen", retain_child)
    pid = dispatch._default_spawn(task, str(workspace), board="import-regression")
    assert pid is not None
    assert len(children) == 1
    child = children[0]
    if custom_bin:
        assert child.args[0] == str(shim)
    try:
        child.wait(timeout=45)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    logs = list(root.rglob(f"{task.id}.log"))
    output = "\n".join(p.read_text(encoding="utf-8") for p in logs)
    assert receipt.exists(), output
    observed = json.loads(receipt.read_text(encoding="utf-8"))
    assert not marker.exists(), (marker.read_text() if marker.exists() else "") + output
    assert child.returncode == 1, output  # deliberate no-credentials guard
    assert "no API keys or providers found" in output, output
    assert "Traceback" not in output, output
    assert observed["bootstrapped"], output
    assert Path(observed["cli_file"]).resolve() == Path(dispatch.__file__).with_name("main.py").resolve()
    import inspect
    assert Path(observed["inspect"]).resolve() == Path(inspect.__file__).resolve()
    assert observed["cwd"] == str(workspace)
    assert observed["terminal_cwd"] == str(workspace)
    assert observed["workspace"] == str(workspace)
    assert observed["home"] == str(profile)
    assert observed["board"] == "import-regression"
    assert observed["task"] == task.id
    expected_env = {"PYTHONPATH": str(observer)}
    if not custom_bin:
        from cron.scheduler_worker_env import pin_hermes_tree_on_pythonpath
        pin_hermes_tree_on_pythonpath(expected_env, Path(dispatch.__file__).resolve().parents[1])
    assert observed["pythonpath"] == expected_env["PYTHONPATH"]
    assert str(workspace) not in observed["sys_path"]
    assert observed["network_attempts"] == [], json.dumps(observed["network_attempts"], indent=2)
