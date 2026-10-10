"""The stop guard follows durable run ownership, not attempted tool calls."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.turn_stop_gates import _kanban_stop_nudge
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from tools import kanban_tools as tools


@pytest.fixture
def worker(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
                "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_STOP_NUDGE", "HERMES_SESSION_ID"):
        monkeypatch.delenv(key, raising=False)
    kb.init_db()
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Verify a worker lifecycle", assignee="default")
        assert kb.claim_task(conn, task_id)
        run_id = kb.get_task(conn, task_id).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return task_id, run_id


@pytest.mark.parametrize("tool_name", [
    "kanban_complete", "kanban_block", "kanban_schedule", "kanban_request_review", "kanban_request_changes",
])
def test_rejected_handoff_does_not_disable_recovery(worker, tool_name):
    assert _kanban_stop_nudge(SimpleNamespace(), []) is not None
    handler = getattr(tools, "_handle_" + tool_name.removeprefix("kanban_"))
    # Schedule permits an omitted reason; use its real invalid-input path.
    args = {"reason": 123} if tool_name == "kanban_schedule" else {}
    result = handler(args)
    assert json.loads(result).get("error")
    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "rejected-handoff", "type": "function",
            "function": {"name": tool_name, "arguments": "{}"},
        }]},
        {"role": "tool", "name": tool_name, "tool_call_id": "rejected-handoff", "content": result},
    ]
    with kbc.connect_closing() as conn:
        assert kb.goal_run_status(conn, *worker) == "running"
    assert _kanban_stop_nudge(SimpleNamespace(), messages) is not None
    # Keep the existing bounded-retry protection; rejection never means an infinite loop.
    assert _kanban_stop_nudge(SimpleNamespace(_kanban_stop_nudges=2), messages) is None


@pytest.mark.parametrize("transition", ["complete", "block", "schedule", "review", "changes"])
def test_committed_handoff_survives_missing_transcript_and_successor(worker, monkeypatch, transition):
    task_id, run_id = worker
    assert _kanban_stop_nudge(SimpleNamespace(), []) is not None
    with kbc.connect_closing() as conn:
        if transition == "complete":
            assert kb.complete_task(conn, task_id, summary="Verified work", expected_run_id=run_id)
        elif transition == "block":
            assert kb.block_task(conn, task_id, reason="Operator decision required", kind="needs_input", expected_run_id=run_id)
        elif transition == "schedule":
            assert kb.schedule_task(conn, task_id, reason="Wait for prerequisite", expected_run_id=run_id)
        else:
            assert kb.request_review(conn, task_id, summary="Verified implementation", expected_run_id=run_id)
            if transition == "changes":
                assert kb.claim_review_task(conn, task_id)
                run_id = kb.get_task(conn, task_id).current_run_id
                monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
                ok, _ = kb.request_changes(conn, task_id, reason="Correct the failing check", expected_run_id=run_id)
                assert ok
                # A new worker is running the card, but the prior review run is finished.
                assert kb.claim_task(conn, task_id)
        assert kb.goal_run_status(conn, task_id, run_id) != "running"
    # CLI/execute_code calls and compacted history need not contain a named terminal tool.
    assert _kanban_stop_nudge(SimpleNamespace(), []) is None


def test_conversation_recovers_rejected_handoff_then_commits(worker):
    """Real agent loop and Kanban handlers; only model responses are scripted."""
    from unittest.mock import MagicMock, patch
    from run_agent import AIAgent
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="test-only-key", base_url="http://127.0.0.1:9/v1",
            model="test-model", provider="custom", quiet_mode=True,
            enabled_toolsets=["kanban"], skip_context_files=True, skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "Complete the assigned test task through the board."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content=None, finish_reason="tool_calls", tool_calls=[
            _mock_tool_call(name="kanban_complete", arguments="{}", call_id="rejected")]),
        _mock_response(content="Work is complete.", finish_reason="stop"),
        _mock_response(content=None, finish_reason="tool_calls", tool_calls=[
            _mock_tool_call(name="kanban_complete", arguments=json.dumps({"summary": "Verified test contract"}), call_id="accepted")]),
        _mock_response(content="Board handoff committed.", finish_reason="stop"),
    ]
    with patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"):
        result = agent.run_conversation("Complete this deterministic lifecycle test.")
    assert result["final_response"] == "Board handoff committed."
    assert result["api_calls"] == 4
    assert agent._kanban_stop_nudges == 1
    tool_rows = [row for row in result["messages"] if row.get("role") == "tool"]
    assert any(json.loads(row["content"]).get("error") for row in tool_rows)
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, worker[0]).status == "done"
        assert kb.goal_run_status(conn, *worker) == "done"
