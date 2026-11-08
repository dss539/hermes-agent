"""Public describe dispatch followed by model-emitted direct call validation."""
import json
from types import SimpleNamespace


def test_describe_then_direct_call_reaches_the_described_tool(monkeypatch):
    import model_tools
    from agent import tool_executor as executor
    from agent.turn_tool_validation import validate_tool_calls
    from tools.registry import registry
    name, toolset = 'delegate_task_to_model', 'describe-direct-probe'
    registry.register(name=name, toolset=toolset,
        schema={'name': name, 'description':'Probe', 'parameters': {
            'type':'object','properties':{'goal':{'type':'string'}},'required':['goal']}},
        handler=lambda args, **kw: json.dumps({'ran':name,'goal':args['goal']}))
    try:
        offered = {'delegate_task', 'tool_search', 'tool_describe', 'tool_call'}
        agent = SimpleNamespace(enabled_toolsets=[toolset, 'delegation'], disabled_toolsets=[],
            valid_tool_names=offered, tools=[{'type':'function','function':{'name':n,'parameters':{}}} for n in sorted(offered)],
            _invalid_tool_retries=0, _invalid_json_retries=0,
            _touch_activity=lambda *a:None, _subdirectory_hints=SimpleNamespace(check_tool_call=lambda *a:None),
            _tool_result_content_for_active_model=lambda n, content:content, tool_progress_callback=None,
            _uniquify_tool_call_ids=lambda *a, **kw:None, _repair_tool_call=lambda n:'delegate_task',
            _buffer_vprint=lambda *a, **kw:None, _vprint=lambda *a, **kw:None, log_prefix='')
        described = model_tools.handle_function_call('tool_describe', {'names':[name]},
                    enabled_toolsets=agent.enabled_toolsets, disabled_toolsets=[])
        assert name in json.loads(described)['tools']
        monkeypatch.setattr(executor, 'get_active_env', lambda *a:None)
        monkeypatch.setattr(executor, 'maybe_persist_tool_result', lambda **kw:'result')
        monkeypatch.setattr(executor, 'note_tool_result', lambda *a:None)
        monkeypatch.setattr(executor, '_flush_session_db_after_tool_progress', lambda *a, **kw:True)
        executor._commit_tool_result(agent, [], executor._ToolCallRef('tool_describe',{'names':[name]},'task','call',[]),
            described, budget=executor.DEFAULT_BUDGET, tool_duration=0, is_error=False, blocked=False, effect_disposition=None)
        call = SimpleNamespace(id='c1', call_id='c1', type='function', function=SimpleNamespace(name=name, arguments='{"goal":"g"}'))
        message = SimpleNamespace(tool_calls=[call])
        verdict = validate_tool_calls(agent, message, 'tool_calls', messages=[], conversation_history=[], api_call_count=1, effective_task_id='task')
        assert verdict.action == 'ok'
        assert message.tool_calls[0].function.name == name
        assert name not in agent.valid_tool_names  # describe returns schemas only; tools[] is unchanged
        assert json.loads(model_tools.handle_function_call(name, {'goal':'g'})) == {'ran':name,'goal':'g'}
    finally:
        registry.deregister(name)
