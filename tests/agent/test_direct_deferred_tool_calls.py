"""A direct call to a Tool Search-deferred tool runs that tool with tool_call's checks.

Models see deferred tool names in the catalog listing and call them directly. The loop
used to treat those names as unknown and fuzzy-repair them to an offered tool (for
example a plugin's ``delegate_task_to_model`` became ``delegate_task``), silently
running a different tool.
"""
import json
from types import SimpleNamespace

import pytest

BRIDGE = {"tool_search", "tool_describe", "tool_call"}


@pytest.fixture
def deferred_tool():
    """Register a plugin-style (hence deferrable) tool whose name is close to an offered one."""
    from tools.registry import registry

    name, toolset = "delegate_task_to_model", "direct-deferred-probe"
    registry.register(
        name=name,
        toolset=toolset,
        schema={"name": name, "description": "Probe", "parameters": {
            "type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}},
        handler=lambda args, **kw: json.dumps({"ran": name, "goal": args.get("goal")}),
    )
    try:
        yield name, toolset
    finally:
        registry.deregister(name)


def _agent(toolsets, offered):
    return SimpleNamespace(
        enabled_toolsets=list(toolsets), disabled_toolsets=[], valid_tool_names=set(offered),
        _invalid_tool_retries=0, _invalid_json_retries=0,
    )


def _call(name, arguments):
    return SimpleNamespace(id="c1", call_id="c1", type="function",
                           function=SimpleNamespace(name=name, arguments=arguments))


def test_direct_deferred_call_is_valid_and_not_repaired(deferred_tool):
    from agent.turn_tool_validation import validate_tool_calls

    name, toolset = deferred_tool
    agent = _agent([toolset, "delegation"], {"delegate_task", *BRIDGE})
    agent._uniquify_tool_call_ids = lambda calls, **kw: None
    agent._repair_tool_call = lambda n: "delegate_task"  # what fuzzy repair returns for this name
    agent._buffer_vprint = agent._vprint = lambda *a, **k: None
    agent.log_prefix = ""
    message = SimpleNamespace(tool_calls=[_call(name, '{"goal": "g"}')])

    verdict = validate_tool_calls(agent, message, "tool_calls", messages=[], conversation_history=[],
                                  api_call_count=1, effective_task_id="t")

    assert verdict.action == "ok"
    assert message.tool_calls[0].function.name == name


def test_direct_deferred_call_gets_bridge_scope_and_argument_checks(deferred_tool):
    from agent.tool_executor import _unwrap_tool_search_call

    name, toolset = deferred_tool
    agent = _agent([toolset], BRIDGE)

    assert _unwrap_tool_search_call(agent, name, {"goal": "g"}) == (name, {"goal": "g"}, None)
    _, _, block = _unwrap_tool_search_call(agent, name, {})
    assert block is not None and "missing required argument(s): goal" in block


def test_deferred_tool_outside_session_scope_stays_invalid(deferred_tool):
    from agent.tool_executor import callable_tool_names

    name, _ = deferred_tool
    assert name not in callable_tool_names(_agent(["delegation"], {"delegate_task", *BRIDGE}))


def test_without_the_tool_call_bridge_deferred_names_are_not_callable(deferred_tool):
    from agent.tool_executor import callable_tool_names

    name, toolset = deferred_tool
    assert callable_tool_names(_agent([toolset], {"terminal"})) == {"terminal"}


def _offered_agent(toolsets, offered):
    agent = _agent(toolsets, offered)
    agent.tools = [{"type": "function", "function": {"name": n, "parameters": {}}} for n in sorted(offered)]
    return agent


def test_tool_describe_loads_the_tool_into_the_tool_list(deferred_tool, monkeypatch):
    """The model calls a deferred tool directly only if it is in the tool list it was sent.
    tool_describe must put it there: appended at the tail (the existing prefix keeps its bytes),
    with the tool's real schema, and pinned so rebuilds and resume keep it."""
    from agent.tool_executor import _load_described_tools
    from tools import mcp_tool_agent

    pinned = []
    monkeypatch.setattr(mcp_tool_agent, "persist_agent_tool_names", lambda agent: pinned.append(agent))
    name, toolset = deferred_tool
    agent = _offered_agent([toolset], BRIDGE)
    before = list(agent.tools)

    _load_described_tools(agent, json.dumps({"tools": {name: {"parameters": {}}}}))

    assert agent.tools[:len(before)] == before
    assert [t["function"]["name"] for t in agent.tools[len(before):]] == [name]
    assert agent.tools[-1]["function"]["parameters"]["required"] == ["goal"]
    assert name in agent.valid_tool_names and pinned == [agent]

    _load_described_tools(agent, json.dumps({"tools": {name: {}}}))  # idempotent
    assert [t["function"]["name"] for t in agent.tools].count(name) == 1


def test_a_loaded_tool_survives_an_agent_rebuild_from_the_session_pin(deferred_tool, monkeypatch):
    """A fresh agent (gateway cache eviction, --resume) is built without the loaded tool; the
    session's tools[] pin must restore it, byte-identical, so the model can keep calling it."""
    from agent.tool_executor import _load_described_tools
    from tools import mcp_tool_agent

    monkeypatch.setattr(mcp_tool_agent, "persist_agent_tool_names", lambda agent: None)
    name, toolset = deferred_tool
    loaded = _offered_agent([toolset], BRIDGE)
    _load_described_tools(loaded, json.dumps({"tools": {name: {}}}))
    pin = {"version": mcp_tool_agent.tool_pin_version(), "tools": list(loaded.tools)}

    fresh = _offered_agent([toolset], BRIDGE)
    assert name not in fresh.valid_tool_names
    mcp_tool_agent.restore_agent_tool_prefix(fresh, pin)
    assert fresh.tools == loaded.tools
    assert name in fresh.valid_tool_names


def test_tool_describe_never_loads_a_tool_outside_session_scope(deferred_tool):
    from agent.tool_executor import _load_described_tools

    name, _ = deferred_tool
    agent = _offered_agent(["delegation"], BRIDGE)
    _load_described_tools(agent, json.dumps({"tools": {name: {}}}))
    assert name not in agent.valid_tool_names
    assert name not in [t["function"]["name"] for t in agent.tools]


def test_session_drops_gate_callable_names_and_loading_and_invalidate_cache(deferred_tool, monkeypatch):
    from agent.tool_executor import callable_tool_names, _load_described_tools
    import toolsets

    name, toolset = deferred_tool
    agent = _offered_agent([toolset], BRIDGE)
    monkeypatch.setattr(toolsets, "agent_tool_drops", lambda a: frozenset({name}) if a.side_agent else frozenset())
    agent.side_agent = False
    assert name in callable_tool_names(agent)
    agent.side_agent = True
    assert name not in callable_tool_names(agent)
    _load_described_tools(agent, json.dumps({"tools": {name: {}}}))
    assert name not in agent.valid_tool_names
    assert all(t["function"]["name"] != name for t in agent.tools)


@pytest.mark.parametrize("is_error,blocked,flushed,loads", [
    (False, False, True, True),
    (True, False, True, False),
    (False, True, True, False),
    (False, False, False, False),
])
def test_describe_completion_loads_only_after_successful_flush(
        deferred_tool, monkeypatch, is_error, blocked, flushed, loads):
    from agent import tool_executor as executor
    from tools import mcp_tool_agent

    name, toolset = deferred_tool
    agent = _offered_agent([toolset], BRIDGE)
    agent._touch_activity = lambda *a: None
    agent._subdirectory_hints = SimpleNamespace(check_tool_call=lambda *a: None)
    agent._tool_result_content_for_active_model = lambda n, content: content
    agent.tool_progress_callback = None
    events = []
    monkeypatch.setattr(executor, "get_active_env", lambda *a: None)
    monkeypatch.setattr(executor, "maybe_persist_tool_result", lambda **kw: "spilled-result-stub")
    monkeypatch.setattr(executor, "note_tool_result", lambda *a: None)

    def flush(a, messages, **kw):
        assert name not in agent.valid_tool_names
        assert messages[-1]["name"] == "tool_describe"
        events.append("flush")
        return flushed

    monkeypatch.setattr(executor, "_flush_session_db_after_tool_progress", flush)
    monkeypatch.setattr(mcp_tool_agent, "persist_agent_tool_names", lambda a: events.append("pin"))
    messages = []
    # Partial describe success and a guardrail suffix must still load the successful entry.
    result = json.dumps({"tools": {name: {}}, "not_found": ["missing"]}) + "\nGuardrail notice"
    committed = executor._commit_tool_result(
        agent, messages, executor._ToolCallRef("tool_describe", {"names": [name]}, "task", "call", []),
        result, budget=executor.DEFAULT_BUDGET, tool_duration=0, is_error=is_error, blocked=blocked,
        effect_disposition=None)
    assert (name in agent.valid_tool_names) is loads
    assert events == (["flush", "pin"] if loads else ["flush"])
    assert (committed is not None) is flushed


def test_mixed_round_keeps_direct_deferred_call_with_expanded_bridge_entries(deferred_tool, monkeypatch):
    from agent import turn_tool_round as round_module
    from agent.tool_executor import _parse_tool_call
    from tools.registry import registry

    name, toolset = deferred_tool
    agent = _offered_agent([toolset], BRIDGE)
    agent.quiet_mode = True
    agent.verbose_logging = False
    agent._uniquify_tool_call_ids = lambda calls, **kw: None
    agent._repair_tool_call = lambda n: None
    agent._buffer_vprint = lambda *a, **kw: None
    agent._cap_delegate_task_calls = agent._deduplicate_tool_calls = lambda calls: calls
    agent._flush_messages_to_session_db = lambda *a: True
    agent._emit_interim_assistant_message = lambda *a: None
    agent.stream_delta_callback = None
    agent._tool_guardrail_halt_decision = None
    agent._touch_activity = lambda *a: None
    executed = []

    def execute(message, messages, *args):
        for tc in message.tool_calls:
            parsed = _parse_tool_call(agent, tc)
            assert parsed.parse_error is None and parsed.scope_block is None
            result = registry.dispatch(parsed.name, parsed.args)
            executed.append((parsed.name, parsed.args["goal"]))
            messages.append({"role": "tool", "name": tc.function.name,
                             "tool_call_id": tc.id, "content": result})

    agent._execute_tool_calls = execute
    direct = _call(name, '{"goal":"direct"}')
    unknown = _call("unknown_probe", "{malformed")
    unknown.id = unknown.call_id = "c2"
    bridge = _call("tool_call", json.dumps({"calls": [
        {"name": name, "arguments": {"goal": "bridge1"}},
        {"name": name, "arguments": {"goal": "bridge2"}},
    ]}))
    bridge.id = bridge.call_id = "c3"
    message = SimpleNamespace(tool_calls=[direct, unknown, bridge])
    monkeypatch.setattr(round_module, "stage_tool_call_message", lambda a, **kw: ({
        "role": "assistant", "content": "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {
                "name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in kw["assistant_message"].tool_calls]}, False))
    monkeypatch.setattr(round_module, "compress_after_tool_results", lambda a, **kw: SimpleNamespace(
        messages=kw["messages"], active_system_prompt=kw["active_system_prompt"],
        conversation_history=kw["conversation_history"], compression_attempts=kw["compression_attempts"],
        final_response=kw["final_response"], turn_exit_reason=kw["turn_exit_reason"],
        current_turn_user_idx=kw["current_turn_user_idx"], end_turn=True))
    messages = []
    verdict = round_module.run_tool_round(
        agent, assistant_message=message, finish_reason="tool_calls", messages=messages,
        conversation_history=[], api_call_count=1, effective_task_id="task", user_message="test",
        system_message={}, active_system_prompt="test", compression_attempts=0,
        max_compression_attempts=1, final_response="", failed=False, _turn_exit_reason="",
        truncated_tool_call_retries=0, current_turn_user_idx=0)
    assert verdict.action == "break"
    assert executed == [(name, "direct"), (name, "bridge1"), (name, "bridge2")]
    staged_ids = [tc["id"] for tc in messages[0]["tool_calls"]]
    result_ids = [row["tool_call_id"] for row in messages[1:]]
    assert len(staged_ids) == len(set(staged_ids)) == 4
    assert sorted(result_ids) == sorted(staged_ids)
    errors = [row for row in messages[1:] if row["name"] == "unknown_probe"]
    assert len(errors) == 1 and "unknown_probe" in errors[0]["content"]


def test_loaded_tool_survives_prefix_refresh_but_drops_on_compaction(deferred_tool, monkeypatch):
    from agent.tool_executor import _load_described_tools
    from tools import mcp_tool_agent

    monkeypatch.setattr(mcp_tool_agent, "persist_agent_tool_names", lambda a: None)
    name, toolset = deferred_tool
    agent = _offered_agent([toolset], BRIDGE)
    _load_described_tools(agent, json.dumps({"tools": {name: {}}}))
    loaded = list(agent.tools)
    mcp_tool_agent.refresh_agent_mcp_tools(agent, preserve_prefix=True)
    assert agent.tools == loaded
    assert name in agent.valid_tool_names
    mcp_tool_agent.refresh_agent_mcp_tools(agent, content_aware=True)
    assert name not in agent.valid_tool_names
    assert all(t["function"]["name"] != name for t in agent.tools)


def test_describe_never_promotes_remote_connectors(deferred_tool):
    from agent.tool_executor import _load_described_tools

    _, toolset = deferred_tool
    agent = _offered_agent([toolset], BRIDGE)
    before = list(agent.tools)
    _load_described_tools(agent, json.dumps({"tools": {"connectors__gmail__SEND_EMAIL": {}}}))
    assert agent.tools == before
    assert agent.valid_tool_names == BRIDGE
