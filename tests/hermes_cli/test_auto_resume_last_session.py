import argparse


def _make_args(**overrides):
    return argparse.Namespace(
        continue_last=overrides.get("continue_last", None),
        resume=overrides.get("resume", None),
        model=overrides.get("model", None),
        provider=overrides.get("provider", None),
        toolsets=overrides.get("toolsets", None),
        skills=overrides.get("skills", None),
        verbose=overrides.get("verbose", False),
        quiet=overrides.get("quiet", False),
        query=overrides.get("query", None),
        worktree=overrides.get("worktree", False),
        checkpoints=overrides.get("checkpoints", False),
        pass_session_id=overrides.get("pass_session_id", False),
        max_turns=overrides.get("max_turns", None),
        yolo=overrides.get("yolo", False),
        source=overrides.get("source", None),
    )


def test_cmd_chat_auto_resumes_last_session_when_enabled(monkeypatch):
    from hermes_cli import main as hm

    captured = {}

    monkeypatch.setattr(hm, "_auto_resume_last_session_enabled", lambda: True)
    monkeypatch.setattr(hm, "_resolve_last_cli_session", lambda: "session-123")
    monkeypatch.setattr(hm, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)

    def fake_cli_main(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("cli.main", fake_cli_main)

    hm.cmd_chat(_make_args())

    assert captured["resume"] == "session-123"


def test_cmd_chat_does_not_override_explicit_resume(monkeypatch):
    from hermes_cli import main as hm

    captured = {}

    monkeypatch.setattr(hm, "_auto_resume_last_session_enabled", lambda: True)
    monkeypatch.setattr(hm, "_resolve_last_cli_session", lambda: "session-123")
    monkeypatch.setattr(hm, "_resolve_session_by_name_or_id", lambda value: f"resolved:{value}")
    monkeypatch.setattr(hm, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)

    def fake_cli_main(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("cli.main", fake_cli_main)

    hm.cmd_chat(_make_args(resume="manual-session"))

    assert captured["resume"] == "resolved:manual-session"