"""Bounded turn-end recovery for dispatcher-owned Kanban runs.

The board's run record is authoritative: attempting a terminal tool is not a
handoff, and a committed handoff may be absent from the retained transcript.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

from agent.delegation_context import owned_kanban_task

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """Only the dispatcher-owned worker may be nudged to hand off its card."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(owned_kanban_task())


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Nudge only an unfinished owned run, never a completed or superseded run.

    ``messages`` is retained for existing callers, but transcript shape cannot
    establish whether a board transition committed (including CLI/tool wrappers).
    """
    if not kanban_stop_nudge_enabled() or attempts >= max_attempts:
        return None

    tid = task_id or owned_kanban_task()
    raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    # The dispatcher pins both identities; without the run we cannot distinguish
    # this worker from a successor that already owns the same card.
    if not raw_run_id:
        return None
    run_id = int(raw_run_id)
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        if kb.goal_run_status(conn, tid, run_id) != "running":
            return None

    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still running under this worker's run. No terminal board "
        "handoff has committed. A rejected tool call does not count. Ending now "
        "causes a protocol violation (clean exit with the card still `running`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work is done "
        "and needs no review, `kanban_request_review(summary=...)` if it is a code "
        "change that needs same-card review, `kanban_block(reason=...)` if you are "
        "blocked, OR `kanban_schedule(reason=...)` for a timed wait. Reviewers approve "
        "with `kanban_complete` or send the card back with "
        "`kanban_request_changes(reason=...)`. Check the tool result and correct any "
        "rejected handoff before stopping.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled"]
