"""Session restrictions apply to catalog reads and both bridge dispatch paths."""
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def scoped_agent(monkeypatch):
    import toolsets
    from tools.registry import registry

    name = "deferred_scope_probe"
    toolset = "deferred-scope-probe"
    registry.register(name=name, toolset=toolset,
        schema={"name": name, "description": "Probe", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: json.dumps({"ran": name}))
    agent = SimpleNamespace(enabled_toolsets=[toolset], disabled_toolsets=[], platform="cli",
        side_agent=False, valid_tool_names={"tool_call", "tool_describe", "tool_search"},
        session_id="scope-probe", _context_engine_tool_names=set(), _memory_manager=None, quiet_mode=False)
    monkeypatch.setattr(toolsets, "agent_tool_drops", lambda a: frozenset({name}) if a.side_agent else frozenset())
    try:
        yield agent, name
    finally:
        registry.deregister(name)


def test_drop_lookup_error_hides_previously_available_deferred_names(scoped_agent, monkeypatch):
    import toolsets
    from agent.tool_executor import _tool_search_scoped_names

    agent, name = scoped_agent
    assert name in _tool_search_scoped_names(agent)

    def unavailable(a):
        raise RuntimeError("session drops unavailable")

    monkeypatch.setattr(toolsets, "agent_tool_drops", unavailable)
    assert _tool_search_scoped_names(agent) == frozenset()


@pytest.mark.parametrize("path", ["sequential", "concurrent"])
def test_bridge_dispatch_and_catalog_obey_session_drop(scoped_agent, path):
    from agent.tool_executor import _resolve_sequential_dispatch, _ToolCallRef
    from agent.agent_runtime_helpers import invoke_tool

    agent, name = scoped_agent

    def call(tool, args):
        if path == "sequential":
            dispatch = _resolve_sequential_dispatch(agent, _ToolCallRef(tool, args, "task", "call", []), [])
            return json.loads(dispatch.execute(args))
        return json.loads(invoke_tool(agent, tool, args, "task", "call", pre_tool_block_checked=True,
            skip_tool_request_middleware=True, skip_tool_execution_middleware=True))

    args = {"name": name, "arguments": {}}
    assert call("tool_call", args) == {"ran": name}
    agent.side_agent = True
    blocked = call("tool_call", args)
    assert "not available in this session" in blocked.get("error", "")
    assert "ran" not in blocked
    described = call("tool_describe", {"names": [name]})
    assert name not in described.get("tools", {})
    assert name in described["not_found"]
