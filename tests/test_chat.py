"""Tests agent/chat.py's wiring without needing a live Anthropic API key or
a real MusicBrainz database — build_options() only constructs config
objects and a DSN string; it never opens a connection or talks to the
Claude API. Actually running the chat loop end-to-end needs both of those
and is deliberately left for manual verification (Stage 5), once a real
ANTHROPIC_API_KEY and an imported musicbrainz database exist.
"""

from agent.chat import SYSTEM_PROMPT, build_options


def test_build_options_wires_semantic_tools(monkeypatch):
    monkeypatch.setenv("PGHOST", "localhost")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "musicbrainz")
    monkeypatch.setenv("MB_RO_USER", "mb_analytics_ro")
    monkeypatch.setenv("MB_RO_PASSWORD", "does-not-need-to-be-real-here")

    options = build_options()

    assert options.system_prompt == SYSTEM_PROMPT
    assert options.tools == []  # no built-in Claude Code tools
    assert set(options.mcp_servers) == {"musicbrainz"}
    assert options.allowed_tools == [
        "mcp__musicbrainz__list_semantic_models",
        "mcp__musicbrainz__describe_model",
        "mcp__musicbrainz__run_semantic_query",
    ]


def test_build_options_requires_env_vars(monkeypatch):
    monkeypatch.delenv("PGHOST", raising=False)
    try:
        build_options()
        raise AssertionError("expected a KeyError for missing PGHOST")
    except KeyError:
        pass
