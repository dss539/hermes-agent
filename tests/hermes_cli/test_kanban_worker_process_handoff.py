"""Offline owner execution boundary: bind a claim, recover a rejected handoff, persist.

The real bind_worker_context/run_worker_turns entry points, agent loop, tool handlers,
SQLite session and board writes run in a private process. Only model responses are
scripted. This is not dispatcher admission, RPC transport, or a running gateway test.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

_CHILD = r'''
import json
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
from gateway import session_kanban as owner

context = json.loads(sys.argv[2])
frame = {'text': 'Complete the assigned lifecycle fixture.', 'policy': {
    'source': 'kanban', 'kanban_json': json.dumps(context),
    'request_json': json.dumps({'turn_v1': {'finite': True, 'unattended': False, 'yolo': False}})},
    'scope': {'execution_id': 'owner-handoff'}, 'route': 'owner-handoff'}
# Bind before tool discovery, just as the managed exec does. No manual worker carrier.
owner.bind_worker_context(frame)
assert os.environ['HERMES_KANBAN_TASK'] == context['task_id']
assert os.environ['HERMES_KANBAN_RUN_ID'] == str(context['run_id'])
assert Path.cwd() == Path(context['workspace'])

from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_run_agent import _mock_response, _mock_tool_call
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing
store = SessionDB(Path(os.environ['HERMES_HOME']) / 'state.db')
with patch('agent.process_bootstrap.OpenAI'):
    agent = AIAgent(api_key='test-only-key', base_url='http://127.0.0.1:9/v1',
        model='test-model', provider='custom', quiet_mode=True,
        enabled_toolsets=['kanban'], skip_context_files=True, skip_memory=True,
        session_id='owner-handoff', session_db=store)
agent.client = MagicMock()
agent._cached_system_prompt = 'Finish the assigned deterministic test task through the board.'
agent._use_prompt_caching = False
agent.compression_enabled = False
agent.save_trajectories = False
responses = iter([
    _mock_response(content=None, finish_reason='tool_calls', tool_calls=[
        _mock_tool_call(name='kanban_complete', arguments='{}', call_id='rejected')]),
    _mock_response(content='Work is complete.', finish_reason='stop'),
    _mock_response(content=None, finish_reason='tool_calls', tool_calls=[
        _mock_tool_call(name='kanban_complete', arguments=json.dumps({'summary': 'Verified process handoff'}), call_id='accepted')]),
    _mock_response(content='Board handoff committed.', finish_reason='stop'),
])
entries = []
def model_response(**kwargs):
    with connect_closing(Path(context['db'])) as conn:
        status = kb.get_task(conn, context['task_id']).status
    entries.append(status)
    if len(entries) == 3:
        assert status == 'running'  # rejected completion did not settle the run
        assert owner.worker_result(context['db'], context) == {}
        assert agent._kanban_stop_nudges == 1
        assert any(m['role'] == 'user' and 'kanban_complete' in str(m.get('content'))
                   and context['task_id'] in str(m.get('content')) for m in kwargs['messages'])
    return next(responses)
agent.client.chat.completions.create.side_effect = model_response
try:
    result = owner.run_worker_turns(agent, frame, [])
    Path(sys.argv[1]).write_text(json.dumps({'final_response': result['final_response'],
        'api_calls': result['api_calls'], 'nudges': getattr(agent, '_kanban_stop_nudges', 0),
        'entries': entries, 'session_id': agent.session_id}))
finally:
    store.close()
'''


def test_owner_process_recovers_rejected_completion(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_state import SessionDB
    from gateway.session_kanban import worker_result

    home, root, workspace = tmp_path / 'home', tmp_path / 'state', tmp_path / 'task'
    for path in (home, root, workspace):
        path.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: home)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setenv('HERMES_KANBAN_DB', str(root / 'kanban.db'))
    kb.init_db()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title='Local owner handoff', assignee='default',
                             workspace_kind='dir', workspace_path=str(workspace))
        assert kb.claim_task(conn, tid)
        task = kb.get_task(conn, tid)
        context = {'db': str(root / 'kanban.db'), 'board': 'default', 'task_id': tid,
            'run_id': task.current_run_id, 'claim_lock': task.claim_lock,
            'workspace': str(workspace), 'workspaces_root': str(tmp_path), 'profile': 'default',
            'branch': None, 'tenant': None, 'goal_mode': False, 'max_runtime_seconds': 60,
            'skills': [], 'context': kb.build_worker_context(conn, tid)}
    import hermes_cli.kanban_db_dispatch as dispatch
    repo = Path(dispatch.__file__).resolve().parent.parent
    env = {
        'HOME': str(home), 'HERMES_HOME': str(root), 'HERMES_TEST_ISOLATION': str(root),
        'TMPDIR': str(tmp_path), 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.defpath,
        'PYTHONPATH': str(repo), 'HERMES_DISABLE_LAZY_INSTALLS': '1', 'HERMES_DISABLE_PLUGINS': '1',
        'AWS_EC2_METADATA_DISABLED': 'true',
    }
    receipt_path = tmp_path / 'receipt.json'
    child = subprocess.run([sys.executable, '-P', '-c', _CHILD, str(receipt_path), json.dumps(context)],
                           cwd=workspace, env=env, capture_output=True, text=True, timeout=60, check=False)
    assert child.returncode == 0, child.stdout + child.stderr
    assert receipt_path.exists(), child.stdout + child.stderr
    receipt = json.loads(receipt_path.read_text())
    assert receipt == {'final_response': 'Board handoff committed.', 'api_calls': 4, 'nudges': 1,
                      'entries': ['running', 'running', 'running', 'done'], 'session_id': 'owner-handoff'}
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == 'done'
        assert kb.goal_run_status(conn, tid, task.current_run_id) == 'done'
        run = conn.execute('SELECT outcome, ended_at FROM task_runs WHERE id=?', (task.current_run_id,)).fetchone()
        assert run['outcome'] == 'completed' and run['ended_at'] is not None
        bound = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='worker_bound'", (tid,)).fetchall()
        assert len(bound) == 1
        assert json.loads(bound[0]['payload'])['claim_lock'] == task.claim_lock
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='worker_result'", (tid,)).fetchone()[0] == 1
    durable = worker_result(context['db'], context)
    assert durable == {'pid': json.loads(bound[0]['payload'])['pid'], 'claim_lock': task.claim_lock,
                       'exit_code': 0, 'last_output': 'Board handoff committed.'}
    # Reopen after the child exits: no in-memory receipt can stand in for durable history.
    store = SessionDB(root / 'state.db')
    try:
        assert store.get_session(receipt['session_id']) is not None
        messages = store.get_messages_as_conversation(receipt['session_id'])
    finally:
        store.close()
    tools = {m['tool_call_id']: json.loads(m['content']) for m in messages if m['role'] == 'tool'}
    assert tools['rejected']['error']
    assert not tools['accepted'].get('error')
    # Stop nudges/candidate prose are intentionally ephemeral (session_persistence.py).
    # The live third model entry above proves the nudge; resume must not replay it.
    assert [m['content'] for m in messages if m['role'] == 'user'] == [
        context['context'] + '\n\n' + 'Complete the assigned lifecycle fixture.']
    assert not any(m.get('content') == 'Work is complete.' for m in messages)
    assert messages[-1]['role'] == 'assistant' and messages[-1]['content'] == receipt['final_response']


@pytest.mark.parametrize('entry_point', ['bind_worker_context', 'run_worker_turns'])
def test_owner_handoff_detects_broken_entry_point(tmp_path, monkeypatch, entry_point):
    prefix = f'''
import gateway.session_kanban as owner

def broken(*args, **kwargs):
    raise AssertionError('OWNER_ENTRY_POINT: {entry_point}')
owner.{entry_point} = broken
'''
    monkeypatch.setattr(sys.modules[__name__], '_CHILD', prefix + _CHILD)
    with pytest.raises(AssertionError, match=f'OWNER_ENTRY_POINT: {entry_point}'):
        test_owner_process_recovers_rejected_completion(tmp_path, monkeypatch)
