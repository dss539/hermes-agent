"""Automatic triage must not undo the repeated-block circuit breaker."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import kanban_watchers_dispatcher as kwd
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp


@pytest.mark.parametrize("kind", ["transient", "capability", "needs_input", None])
def test_auto_decompose_preserves_escalations_and_processes_new_work(tmp_path, monkeypatch, kind):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kwd, "_board_slugs", lambda _kb: ["default"])
    kb.init_db()
    with kbc.connect_closing() as conn:
        escalated = kb.create_task(conn, title="Waiting for a real prerequisite", assignee="default")
        kb.recompute_ready(conn)
        assert kb.block_task(conn, escalated, reason="Unresolved prerequisite", kind=kind)
        assert kb.unblock_task(conn, escalated)
        assert kb.block_task(conn, escalated, reason="Same unresolved prerequisite", kind=kind)
        assert kb.get_task(conn, escalated).status == "triage"
        fresh = kb.create_task(conn, title="New triage request", triage=True)

    calls = []
    def scripted_aux(_operation, task_id, **kwargs):
        calls.append(task_id)
        return json.dumps({"fanout": False, "title": "Specified new work", "body": "Explicit acceptance criteria"}), ""

    monkeypatch.setattr(decomp, "_call_aux", scripted_aux)
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    dispatcher = kwd._KanbanDispatcher(kb, settings)
    assert dispatcher.auto_decompose_tick(5) == 1
    assert calls == [fresh]
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, escalated)
        assert task.status == "triage"
        assert task.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT
        assert kb.get_task(conn, fresh).status == "ready"
    assert dispatcher.auto_decompose_tick(5) == 0
    assert calls == [fresh]

    # Explicit supervisor recovery is not disabled by the automatic admission gate.
    assert escalated in decomp.list_triage_ids()
    assert decomp.decompose_task(escalated, author="supervisor").ok
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, escalated).status == "ready"


def test_escalated_page_cannot_hide_fresh_intake(tmp_path, monkeypatch):
    home = tmp_path / "isolated-state"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as conn:
        # Bulk fixture: a whole higher-priority page of already-escalated cards.
        # The small lifecycle test above establishes how these states are reached.
        conn.executemany(
            "INSERT INTO tasks (id, title, status, priority, created_at, "
            "workspace_kind, block_recurrences, tenant) VALUES (?, ?, 'triage', 10, 1, 'scratch', ?, 'one')",
            [(f"t_escalated_{i}", "Unresolved prerequisite", kb.BLOCK_RECURRENCE_LIMIT) for i in range(1000)],
        )
        conn.commit()
        fresh = kb.create_task(conn, title="New work", tenant="one", triage=True)
        other = kb.create_task(conn, title="Other tenant", tenant="two", priority=20, triage=True)
    assert decomp.list_triage_ids(automatic=True, tenant="one") == [fresh]
    assert decomp.list_triage_ids(automatic=True) == [other, fresh]
    assert decomp.list_triage_ids(automatic=True, tenant="absent") == []
    manual = decomp.list_triage_ids(tenant="one")
    assert len(manual) == 1000
    assert fresh not in manual  # preserve the existing unfiltered manual window
    assert set(manual) == {f"t_escalated_{i}" for i in range(1000)}
