"""A separate owner execution process recovers a rejected handoff against real SQLite.

The dispatcher now submits to an owner instead of starting a one-shot CLI agent.
This exercises that owner's real agent phase without starting a gateway.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

_CHILD = r'''
import json
from unittest.mock import MagicMock, patch
from run_agent import AIAgent
from tests.agent.test_run_agent import _mock_response, _mock_tool_call
with patch('agent.process_bootstrap.OpenAI'):
    agent = AIAgent(api_key='test-only-key', base_url='http://127.0.0.1:9/v1',
        model='test-model', provider='custom', quiet_mode=True,
        enabled_toolsets=['kanban'], skip_context_files=True, skip_memory=True)
agent.client = MagicMock()
agent._cached_system_prompt = 'Finish the assigned deterministic test task through the board.'
agent._use_prompt_caching = False
agent.compression_enabled = False
agent.save_trajectories = False
agent.client.chat.completions.create.side_effect = [
    _mock_response(content=None, finish_reason='tool_calls', tool_calls=[
        _mock_tool_call(name='kanban_complete', arguments='{}', call_id='rejected')]),
    _mock_response(content='Work is complete.', finish_reason='stop'),
    _mock_response(content=None, finish_reason='tool_calls', tool_calls=[
        _mock_tool_call(name='kanban_complete', arguments=json.dumps({'summary': 'Verified process handoff'}), call_id='accepted')]),
    _mock_response(content='Board handoff committed.', finish_reason='stop'),
]
with patch.object(agent, '_persist_session'), patch.object(agent, '_save_trajectory'):
    result = agent.run_conversation('Complete the assigned lifecycle fixture.')
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(json.dumps({'final_response': result['final_response'], 'api_calls': result['api_calls'],
                  'nudges': getattr(agent, '_kanban_stop_nudges', 0)}))
'''


def test_owner_process_recovers_rejected_completion(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    home, root, workspace = tmp_path / 'home', tmp_path / 'state', tmp_path / 'task'
    for path in (home, root, workspace):
        path.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: home)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setenv('HERMES_KANBAN_DB', str(root / 'kanban.db'))
    kb.init_db()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title='Local process handoff', assignee='default',
                             workspace_kind='dir', workspace_path=str(workspace))
        assert kb.claim_task(conn, tid)
        task = kb.get_task(conn, tid)
    import hermes_cli.kanban_db_dispatch as dispatch
    repo = Path(dispatch.__file__).resolve().parent.parent
    env = {
        'HOME': str(home), 'HERMES_HOME': str(root), 'HERMES_TEST_ISOLATION': str(root),
        'TMPDIR': str(tmp_path), 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.defpath,
        'PYTHONPATH': str(repo), 'HERMES_DISABLE_LAZY_INSTALLS': '1', 'HERMES_DISABLE_PLUGINS': '1',
        'HERMES_KANBAN_DB': str(root / 'kanban.db'), 'HERMES_KANBAN_TASK': tid,
        'HERMES_KANBAN_RUN_ID': str(task.current_run_id), 'HERMES_KANBAN_CLAIM_LOCK': task.claim_lock,
        'AWS_EC2_METADATA_DISABLED': 'true',
    }
    receipt_path = tmp_path / 'receipt.json'
    child = subprocess.run([sys.executable, '-P', '-c', _CHILD, str(receipt_path)], cwd=workspace, env=env,
                           capture_output=True, text=True, timeout=60, check=False)
    assert child.returncode == 0, child.stdout + child.stderr
    assert receipt_path.exists(), child.stdout + child.stderr
    receipt = json.loads(receipt_path.read_text())
    assert receipt == {'final_response': 'Board handoff committed.', 'api_calls': 4, 'nudges': 1}
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == 'done'
        assert kb.goal_run_status(conn, tid, task.current_run_id) == 'done'
        run = conn.execute('SELECT outcome FROM task_runs WHERE id=?', (task.current_run_id,)).fetchone()
        assert run['outcome'] == 'completed'
