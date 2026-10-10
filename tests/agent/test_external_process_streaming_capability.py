"""External-process streaming is opt-in without bypassing transport safeguards."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.turn_api_call import _should_stream
from providers.base import ProviderProfile


def _agent(provider="streaming-test-process", *, consumers=True, **overrides):
    fields = dict(
        provider=provider,
        base_url="https://provider.example.invalid/v1",
        client=object(),
        _has_stream_consumers=lambda: consumers,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def process_profile(monkeypatch):
    profile = ProviderProfile(
        name="streaming-test-process", auth_type="external_process"
    )
    monkeypatch.setattr(
        "providers.get_provider_profile",
        lambda name: profile if name == profile.name else None,
    )
    return profile


@pytest.mark.parametrize("consumers", [False, True])
def test_external_process_explicit_opt_in_streams(process_profile, consumers):
    # Assignment also works on the pre-capability host: the regression must
    # fail at the streaming decision, not at an unknown constructor keyword.
    process_profile.supports_streaming = True
    assert _should_stream(_agent(consumers=consumers)) is True


def test_external_process_defaults_to_nonstreaming(process_profile):
    assert _should_stream(_agent()) is False


@pytest.mark.parametrize("capability", [False, None, 0, 1, "true", "false", Mock()])
def test_external_process_requires_boolean_true(process_profile, capability):
    process_profile.supports_streaming = capability
    assert _should_stream(_agent()) is False


@pytest.mark.parametrize(
    "base_url",
    ["acp://process", "acp+tcp://127.0.0.1:9000", "ACP://process", "ACP+TCP://localhost:9000"],
)
def test_acp_scheme_refuses_even_opted_in_streaming(process_profile, base_url):
    process_profile.supports_streaming = True
    assert _should_stream(_agent(base_url=base_url)) is False


@pytest.mark.parametrize("provider", ["streaming-test-process", "streaming-test-http"])
def test_explicit_disable_wins_over_provider_defaults(process_profile, provider):
    process_profile.supports_streaming = True
    assert _should_stream(_agent(provider, _disable_streaming=True)) is False


@pytest.mark.parametrize("consumers", [False, True])
def test_http_provider_streams_with_default_profile(monkeypatch, consumers):
    profile = ProviderProfile(name="streaming-test-http")
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    assert _should_stream(_agent(profile.name, consumers=consumers)) is True


@pytest.mark.parametrize("consumers", [False, True])
def test_mock_client_guard_survives_external_process_opt_in(process_profile, consumers):
    process_profile.supports_streaming = True
    assert _should_stream(_agent(consumers=consumers, client=Mock())) is consumers


@pytest.mark.parametrize("consumers", [False, True])
def test_moa_without_display_consumer_stays_nonstreaming(process_profile, consumers):
    assert _should_stream(_agent("moa", consumers=consumers)) is consumers


def test_provider_profile_declares_opt_in_default():
    assert ProviderProfile(name="default").supports_streaming is False
    assert ProviderProfile(name="opted-in", supports_streaming=True).supports_streaming is True
