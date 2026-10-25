"""Only explicit stream rejection changes channel; unrelated upstream errors propagate."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


_REPORTED_ERROR = (
    'Incomplete upstream response (first upstream attempt: status 400, capture incomplete, '
    'native retries denied: 1, upstream said: "thinking.type.disabled" is not supported for '
    'this model. Use "thinking.type.adaptive" and "output_config.effort" to control thinking '
    'behavior.): API Error: 400 HERMES_MODEL_ADMISSION_CONSUMED'
)
_EVENT_ORDER_ERROR = 'Unexpected event order, got content_block_delta before "message_start"'
_CASES = [
    pytest.param("Claude upstream error: thinking.type.disabled is not supported", False,
                 id="claude-upstream-thinking-rejection"),
    pytest.param(_REPORTED_ERROR, False, id="reported-upstream-thinking-rejection"),
    pytest.param("Upstream model is not supported", False, id="upstream"),
    pytest.param("Downstream model is not supported", False, id="downstream"),
    pytest.param("Mainstream model is not supported", False, id="mainstream"),
    pytest.param("Streamer is not supported", False, id="streamer"),
    pytest.param("stream_options is not supported", False, id="stream-options"),
    pytest.param("input_streaming is not supported", False, id="input-streaming"),
    pytest.param("Stream failed: thinking.type.disabled is not supported", False,
                 id="unrelated-stream-clause"),
    pytest.param("Streaming response failed because thinking.type.disabled is not supported", False,
                 id="unrelated-streaming-clause"),
    pytest.param("Parameter 'stream' is not supported", True, id="quoted-parameter"),
    pytest.param("Streaming is not supported for this model", True, id="streaming"),
    pytest.param("stream is not supported by this provider", True, id="stream"),
    pytest.param("stream not supported", True, id="stream-no-copula"),
    pytest.param("Error: 'STREAMING' is not supported", True, id="quoted-mixed-case"),
    pytest.param('"STREAMING" is not supported', True, id="double-quoted-mixed-case"),
    pytest.param("Streaming requests are not supported for this model", True, id="streaming-requests"),
    pytest.param("Streaming responses are not supported", True, id="streaming-responses"),
    pytest.param("Streaming is currently not supported by this provider", True, id="streaming-currently"),
    pytest.param("Streams are not supported", True, id="streams-plural"),
    pytest.param("stream: not supported", True, id="stream-colon"),
    pytest.param("Streaming mode is not supported", True, id="streaming-mode"),
    pytest.param("Streaming output is currently not supported", True, id="streaming-output"),
    pytest.param("Non-streaming is not supported for this model", False, id="non-streaming"),
    pytest.param("non-stream is not supported", False, id="non-stream"),
    pytest.param("non streaming is not supported", False, id="non-space-streaming"),
    pytest.param("Non-streams are not supported", False, id="non-streams-plural"),
    pytest.param("Non streaming requests are not supported", False, id="non-streaming-requests"),
    pytest.param("Streaming requests responses are not supported", False, id="multiple-streaming-nouns"),
    pytest.param("Streaming features are not supported", False, id="unlisted-streaming-noun"),
    pytest.param("Streaming is sometimes not supported", False, id="unlisted-streaming-adverb"),
    pytest.param("stream is not supportedness", False, id="support-word-boundary"),
    pytest.param("bedrock:InvokeModelWithResponseStream is not authorized", True, id="bedrock-iam"),
    pytest.param("AccessDeniedException: bedrock:InvokeModelWithResponseStream", True,
                 id="bedrock-access-denied"),
    pytest.param("bedrock:InvokeModelWithResponseStream timed out", False, id="bedrock-timeout"),
    pytest.param("AccessDeniedException: bedrock:InvokeModel", False, id="bedrock-nonstream-denial"),
]


def _streaming_call(api_mode="anthropic_messages", provider="custom"):
    from agent.chat_completion_helpers import _StreamingCall

    # Exercise the real session-state transition without starting unrelated stream workers.
    call = _StreamingCall.__new__(_StreamingCall)
    call.agent = SimpleNamespace(
        api_mode=api_mode, provider=provider, _disable_streaming=False, _safe_print=MagicMock(),
    )
    return call


@pytest.mark.parametrize("message,unavailable", _CASES)
def test_main_stream_rejection_controls_session_flag(message, unavailable):
    call = _streaming_call()
    call._maybe_disable_streaming(RuntimeError(message))

    assert call.agent._disable_streaming is unavailable
    assert call.agent._safe_print.call_count == int(unavailable)


@pytest.mark.parametrize("message,unavailable", _CASES + [
    pytest.param(_EVENT_ORDER_ERROR, True, id="unexpected-event-order"),
])
def test_aux_stream_rejection_controls_create_fallback(message, unavailable):
    from agent.anthropic_adapter import create_anthropic_message

    error = RuntimeError(message)
    client = MagicMock()
    client.messages.stream.side_effect = error
    final_message = SimpleNamespace(content=[])
    client.messages.create.return_value = final_message
    kwargs = {"model": "test-model", "messages": [], "max_tokens": 16}

    if unavailable:
        assert create_anthropic_message(client, kwargs) is final_message
        client.messages.create.assert_called_once_with(**kwargs)
    else:
        with pytest.raises(RuntimeError) as raised:
            create_anthropic_message(client, kwargs)
        assert raised.value is error
        client.messages.create.assert_not_called()
    client.messages.stream.assert_called_once_with(**kwargs)


@pytest.mark.parametrize("message", [
    _EVENT_ORDER_ERROR,
    _EVENT_ORDER_ERROR + "; Claude upstream error: thinking.type.disabled is not supported",
])
@pytest.mark.parametrize("api_mode,provider,disabled", [
    ("anthropic_messages", "custom", True),
    ("anthropic_messages", "bedrock", False),
    ("chat_completions", "custom", False),
])
def test_main_event_order_preserves_provider_specific_recovery(message, api_mode, provider, disabled):
    call = _streaming_call(api_mode, provider)
    call._maybe_disable_streaming(RuntimeError(message))

    assert call.agent._disable_streaming is disabled
    assert call.agent._safe_print.call_count == int(disabled)
    if disabled:
        assert "events out of order" in call.agent._safe_print.call_args.args[0]
