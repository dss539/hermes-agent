import io
from contextlib import redirect_stdout
from unittest.mock import patch


def _make_cli(**kwargs):
    import cli as _cli_mod
    from cli import HermesCLI

    clean_config = {
        "model": {
            "default": "gpt-5.4",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "provider": "openai-codex",
        },
        "display": {"compact": False, "tool_progress": "all", "resume_display": "full"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }

    with (
        patch("cli.get_tool_definitions", return_value=[]),
        patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": clean_config}),
        patch.dict("os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}, clear=False),
    ):
        return HermesCLI(**kwargs)


def test_show_model_and_providers_uses_grouped_layout():
    cli = _make_cli(provider="openai-codex", model="gpt-5.4")

    providers = [
        {
            "slug": "openai-codex",
            "name": "OpenAI Codex",
            "is_current": True,
            "models": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"],
            "total_models": 3,
            "api_url": "https://chatgpt.com/backend-api/codex",
        }
    ]

    out = io.StringIO()
    with (
        patch("hermes_cli.model_switch.list_authenticated_providers", return_value=providers),
        patch("hermes_cli.models.list_available_providers", return_value=[]),
        patch("hermes_cli.auth.resolve_provider", return_value="openai-codex"),
        redirect_stdout(out),
    ):
        cli._show_model_and_providers()

    text = out.getvalue()
    assert "OpenAI Codex [openai-codex] ← active:" in text
    assert "      gpt-5.4" in text
    assert "      gpt-5.4-mini" in text
    assert "      gpt-5.3-codex" in text
    assert "(+" not in text


def test_model_help_listing_groups_ollama_cloud_models():
    cli = _make_cli(provider="openai-codex", model="gpt-5.4")

    providers = [
        {
            "slug": "custom:ollama-gemma4-31b",
            "name": "ollama-gemma4-31b",
            "is_current": False,
            "models": ["gemma4:31b"],
            "total_models": 1,
            "source": "custom-provider",
            "api_url": "https://ollama.com/v1",
        },
        {
            "slug": "custom:ollama-glm-5",
            "name": "ollama-glm-5",
            "is_current": False,
            "models": ["glm-5"],
            "total_models": 1,
            "source": "custom-provider",
            "api_url": "https://ollama.com/v1",
        },
    ]

    out = io.StringIO()
    with (
        patch("hermes_cli.model_switch.list_authenticated_providers", return_value=providers),
        patch("hermes_cli.model_switch.MODEL_ALIASES", {"fast": "gpt-5.4-mini"}),
        patch("cli._cprint", lambda msg: print(msg)),
        redirect_stdout(out),
    ):
        cli._handle_model_switch("/model")

    text = out.getvalue()
    assert "Ollama Cloud [--provider custom:ollama-cloud]:" in text
    assert "    gemma4:31b" in text
    assert "    glm-5" in text
    assert "ollama-gemma4-31b [--provider custom:ollama-gemma4-31b]:" not in text
    assert "ollama-glm-5 [--provider custom:ollama-glm-5]:" not in text
    assert "(+" not in text


def test_show_model_and_providers_groups_ollama_cloud_models():
    cli = _make_cli(provider="openai-codex", model="gpt-5.4")

    providers = [
        {
            "slug": "custom:ollama-gemma4-31b",
            "name": "ollama-gemma4-31b",
            "is_current": False,
            "models": ["gemma4:31b"],
            "total_models": 1,
            "source": "custom-provider",
            "api_url": "https://ollama.com/v1",
        },
        {
            "slug": "custom:ollama-glm-5",
            "name": "ollama-glm-5",
            "is_current": False,
            "models": ["glm-5"],
            "total_models": 1,
            "source": "custom-provider",
            "api_url": "https://ollama.com/v1",
        },
    ]

    out = io.StringIO()
    with (
        patch("hermes_cli.model_switch.list_authenticated_providers", return_value=providers),
        patch("hermes_cli.models.list_available_providers", return_value=[]),
        patch("hermes_cli.auth.resolve_provider", return_value="openai-codex"),
        redirect_stdout(out),
    ):
        cli._show_model_and_providers()

    text = out.getvalue()
    assert "Ollama Cloud [custom:ollama-cloud]:" in text
    assert "      gemma4:31b" in text
    assert "      glm-5" in text
    assert "ollama-gemma4-31b [custom:ollama-gemma4-31b]:" not in text
    assert "ollama-glm-5 [custom:ollama-glm-5]:" not in text
