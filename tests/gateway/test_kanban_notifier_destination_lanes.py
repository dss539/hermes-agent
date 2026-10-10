"""A wake runs the woken agent's whole turn, so one busy destination must not hold
up notifications to other destinations, while each destination still receives its
own events in order, one delivery at a time."""

import asyncio

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self):
        self._host = "127.0.0.1"
        self._port = 8642
        self._api_key = "k"
        self._model_name = "hermes"

    async def send(self, chat_id, text, metadata=None):
        return SendResult(success=False, error="API server uses HTTP request/response, not send()")


def _completed_task_for(chat_id, title):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title=title, assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="api_server", chat_id=chat_id)
        kb.complete_task(conn, tid, summary=f"{title} done")
        return tid
    finally:
        conn.close()


def _unseen(tid, chat_id):
    conn = kbc.connect()
    try:
        _, events = kbn.unseen_events_for_sub(
            conn, task_id=tid, platform="api_server", chat_id=chat_id, kinds=["completed"])
        return events
    finally:
        conn.close()


@pytest.mark.parametrize("same_tick", [False, True])
def test_busy_destination_does_not_delay_others_and_keeps_its_own_order(tmp_path, monkeypatch, same_tick):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "lanes.db"))
    kb.init_db()
    first_a = _completed_task_for("chat-a", "first for a")
    queued_a = _completed_task_for("chat-a", "queued for a") if same_tick else None
    only_b = _completed_task_for("chat-b", "only for b")

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *args, **kwargs):
        await real_sleep(0 if delay == 5 else min(delay, 0.01))

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    posts = []

    async def run():
        release_a = asyncio.Event()

        async def fake_self_post(adapter, *, text, session_id, **kwargs):
            posts.append((session_id, text))
            if session_id == "chat-a":
                await release_a.wait()  # chat-a's woken turn is still running

        import gateway.wake as wake_mod
        monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._running = True
        runner.adapters = {Platform.API_SERVER: ApiServerLikeAdapter()}
        runner._kanban_sub_fail_counts = {}
        runner._kanban_dispatcher_lock_handle = object()

        async def until(predicate):
            for _ in range(500):
                if predicate():
                    return True
                await real_sleep(0.01)
            return False

        watcher = asyncio.create_task(runner._kanban_notifier_watcher(interval=1))
        try:
            assert await until(lambda: any(s == "chat-b" for s, _ in posts)), (
                "chat-b's wake waited behind chat-a's running turn")
            assert await until(lambda: not _unseen(only_b, "chat-b"))
            assert [s for s, _ in posts].count("chat-a") == 1

            # A new event for chat-a while its lane is busy stays unclaimed until the lane ends.
            second_a = _completed_task_for("chat-a", "second for a")
            await real_sleep(0.2)
            assert [s for s, _ in posts].count("chat-a") == 1
            assert _unseen(second_a, "chat-a")

            release_a.set()
            expected_a = [first_a, queued_a, second_a] if same_tick else [first_a, second_a]
            assert await until(lambda: [s for s, _ in posts].count("chat-a") == len(expected_a))
            assert await until(lambda: all(not _unseen(tid, "chat-a") for tid in expected_a))
            a_texts = [t for s, t in posts if s == "chat-a"]
            assert all(tid in text for tid, text in zip(expected_a, a_texts, strict=True))
        finally:
            release_a.set()
            runner._running = False
            await asyncio.wait_for(watcher, timeout=5)

    asyncio.run(run())
