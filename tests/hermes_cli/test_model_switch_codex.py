"""Regression tests for switching from custom endpoints back to OpenAI Codex."""

from hermes_cli.model_switch import switch_model
from hermes_cli.models import detect_provider_for_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def test_detect_provider_for_model_prefers_codex_oauth(monkeypatch):
    """OAuth-backed Codex creds should beat OpenRouter fallback for GPT-5 models."""
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda **kwargs: {"api_key": "codex-token"},
    )

    detected = detect_provider_for_model("gpt-5.4", "anthropic")

    assert detected == ("openai-codex", "gpt-5.4")


def test_switch_model_from_custom_to_gpt5_prefers_codex(monkeypatch):
    """Leaving a custom endpoint for GPT-5 should switch to Codex when the custom endpoint does not list it."""
    monkeypatch.setattr(
        "hermes_cli.models.probe_api_models",
        lambda api_key, base_url, timeout=5.0: {
            "models": ["glm-5.1"],
            "probed_url": f"{base_url}/models",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda model_name, current_provider: ("openai-codex", "gpt-5.4"),
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested: {
            "api_key": "codex-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *args, **kwargs: _MOCK_VALIDATION,
    )
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_capabilities",
        lambda *args, **kwargs: None,
    )

    result = switch_model(
        raw_input="gpt-5.4",
        current_provider="custom",
        current_model="glm-5.1",
        current_base_url="https://ollama.com/api",
        current_api_key="ollama-token",
    )

    assert result.success is True
    assert result.target_provider == "openai-codex"
    assert result.new_model == "gpt-5.4"
    assert result.base_url == "https://chatgpt.com/backend-api/codex"
    assert result.api_mode == "codex_responses"


def test_switch_model_keeps_custom_endpoint_when_model_exists(monkeypatch):
    """If the current custom endpoint lists the requested model, keep the live endpoint and creds."""
    detect_calls = []

    monkeypatch.setattr(
        "hermes_cli.models.probe_api_models",
        lambda api_key, base_url, timeout=5.0: {
            "models": ["glm-5.1", "glm-5.4"],
            "probed_url": f"{base_url}/models",
        },
    )

    def _detect_provider(model_name, current_provider):
        detect_calls.append((model_name, current_provider))
        return ("openai-codex", "gpt-5.4")

    monkeypatch.setattr("hermes_cli.models.detect_provider_for_model", _detect_provider)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested: {
            "api_key": "wrong-token",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *args, **kwargs: _MOCK_VALIDATION,
    )
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_capabilities",
        lambda *args, **kwargs: None,
    )

    result = switch_model(
        raw_input="glm-5.4",
        current_provider="custom",
        current_model="glm-5.1",
        current_base_url="https://ollama.com/api",
        current_api_key="ollama-token",
    )

    assert result.success is True
    assert result.target_provider == "custom"
    assert result.new_model == "glm-5.4"
    assert result.base_url == "https://ollama.com/api"
    assert result.api_key == "ollama-token"
    assert detect_calls == []
