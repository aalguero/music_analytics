# Project State

_Last updated: 2026-08-15_

## What this is

Conversational analytics over the MusicBrainz database via an Apache Ossie semantic layer.
Natural-language questions in, text/insight answers out (no SQL, no visualization yet).
See `README.md` for full setup/architecture.

## Current status: working

`python -m agent.chat` runs end-to-end against the local Postgres MusicBrainz copy.

## Recent changes (this session)

- **`agent/chat.py` now accepts `--model`** (default `claude-sonnet-5`). Previously the model
  was unset, which silently fell back to whatever the `claude` CLI's account-wide default was
  (Opus 5) — an expensive choice for what's really a narrow, mechanical 3-tool workload.
  `README.md` documents the flag.
- **Fixed a real bug found while debugging a "hang"**: the project's `ANTHROPIC_API_KEY` in
  `.env` was invalid (401 `authentication_failed`). The SDK retries auth failures silently up
  to 10x with growing backoff, and `chat.py`'s `_print_message` only handles `AssistantMessage`/
  `ResultMessage` — so an auth failure looked exactly like a frozen terminal instead of an error.
  Key was rotated via console.anthropic.com and updated in `.env`.

## Model comparison notes (informal, from live testing)

- Haiku 4.5 is noticeably cheaper than Sonnet 5.
- Haiku answers can read as *more detailed* than Sonnet's — likely because it relays raw tool
  output more literally rather than condensing/synthesizing it, not because it's more capable.
  Worth spot-checking correctness (not just verbosity) before trusting Haiku answers at face value.
- No default model change was made — user prefers switching via `--model` per session rather
  than picking one fixed default.

## Known open items / ideas discussed, not yet started

1. **Generative BI / visualization** — README already flags this as "planned second phase";
   currently text/insight answers only.
2. **Expand the semantic layer** — curated (`semantic/core/`) currently covers artist, release,
   recording, genre, label, work, release_group. Everything else in the ~180-table schema is
   auto-generated (`semantic/generated/`, gitignored, regenerate via
   `python -m semantic.generator.introspect generate`).
3. **Surface `SystemMessage` events in `chat.py`** — right now retries/errors from the SDK are
   silently dropped by `_print_message`, which is what made the auth failure above look like a
   hang. Printing `api_retry` / error `SystemMessage`s would make future failures visible
   immediately instead of looking frozen.

## Repo housekeeping note

Nothing in this repo is committed to git yet (`git status` shows everything as untracked,
including `README.md`, `agent/`, `semantic/`, `scripts/`, `tests/`, `pyproject.toml`). Worth an
initial commit when ready.
