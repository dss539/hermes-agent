"""Dispatcher's real owner-client process must not import task-local Python."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

_OBSERVER = r'''
import atexit, os, sys
# This fixture must not auto-start or contact any owner/gateway.
os.environ['HERMES_TUI_GATEWAY_URL'] = 'ws://127.0.0.1:1'
def no_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise RuntimeError('offline import fixture forbids network')
sys.addaudithook(no_network)
def report():
    import json
    with open(os.environ['KANBAN_IMPORT_RECEIPT'], 'w') as f:
        json.dump({'inspect': getattr(sys.modules.get('inspect'), '__file__', None),
                   'cwd': os.getcwd(), 'sys_path': sys.path,
                   'pythonpath': os.environ.get('PYTHONPATH')}, f)
atexit.register(report)
'''


@pytest.mark.parametrize('path_entry', [None, '', '.', 'absolute', 'alias'])
def test_worker_client_ignores_task_local_inspect(tmp_path, monkeypatch, path_entry):
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    home, root = tmp_path / 'home', tmp_path / 'state'
    home.mkdir()
    root.mkdir()
    root.joinpath('config.yaml').write_text('nous:\n  guest: false\n')
    observer = tmp_path / 'observer'
    observer.mkdir()
    observer.joinpath('sitecustomize.py').write_text(_OBSERVER)
    workspace = tmp_path / 'task'
    workspace.mkdir()
    marker = workspace / 'inspect-imported'
    workspace.joinpath('inspect.py').write_text(f"open({str(marker)!r}, 'w').write('shadowed')\n")
    if path_entry == 'absolute':
        path_entry = str(workspace)
    elif path_entry == 'alias':
        alias = tmp_path / 'alias'
        alias.symlink_to(workspace, target_is_directory=True)
        path_entry = str(alias)
    receipt = tmp_path / 'receipt.json'
    pp = str(observer) + (os.pathsep + path_entry if path_entry is not None else '')
    monkeypatch.setattr(os, 'environ', {
        'HOME': str(home), 'HERMES_HOME': str(root),
        'HERMES_TEST_ISOLATION': str(root), 'HERMES_DISABLE_LAZY_INSTALLS': '1',
        'HERMES_DISABLE_PLUGINS': '1', 'AWS_EC2_METADATA_DISABLED': 'true',
        'PATH': str(Path(sys.executable).parent), 'PYTHONPATH': pp,
        'KANBAN_IMPORT_RECEIPT': str(receipt),
    })
    task = kb.Task(id='t_import', title='Offline worker import test', body=None,
                   assignee='default', status='running', priority=0, created_by='test',
                   created_at=1, started_at=None, completed_at=None, workspace_kind='dir',
                   workspace_path=str(workspace), claim_lock='lock', claim_expires=None,
                   tenant=None, current_run_id=1)
    children = []
    real_popen = subprocess.Popen
    def retain_child(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, 'Popen', retain_child)
    assert dispatch._default_spawn(task, str(workspace), board='imports') is not None
    assert len(children) == 1
    child = children[0]
    try:
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    logs = '\n'.join(p.read_text() for p in root.rglob('t_import.log'))
    assert receipt.exists(), logs
    observed = json.loads(receipt.read_text())
    assert not marker.exists(), logs
    import inspect
    assert Path(observed['inspect']).resolve() == Path(inspect.__file__).resolve()
    assert observed['cwd'] == str(workspace)
    assert str(workspace) not in observed['sys_path']
    entries = observed['pythonpath'].split(os.pathsep)
    assert str(observer) in entries
    assert all(Path(os.path.join(workspace, entry)).resolve() != workspace.resolve() for entry in entries)
    assert 'Kanban requires the assigned local profile owner' in logs
    assert child.returncode != 0  # deliberate refusal, not a boot/import error
