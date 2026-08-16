"""CLI chat loop: natural-language questions in, answers grounded in the
Ossie semantic layer out. Run with `python -m agent.chat` (add `--model` to
override the default `claude-sonnet-5` — e.g. `--model claude-haiku-4-5-20251001`
for cheaper/faster, or `--model claude-opus-5` for tougher questions). Without
an explicit model, the SDK falls back to whatever the `claude` CLI is
currently configured to use account-wide, which can silently be an expensive
model for what is a narrow, mechanical tool-calling workload — so `chat.py`
always passes one explicitly.

The agent's *only* tools are the three defined in tools.py — no Bash, no
Read/Write, no arbitrary MCP servers (`tools=[]` below disables every
built-in Claude Code tool). It cannot see a table name or write SQL itself;
it can only ask the semantic layer questions in terms of datasets, fields,
and metrics, and sql_builder.py does the actual, deterministic translation
to SQL. See the module docstrings in sql_builder.py and tools.py for why.
"""

from __future__ import annotations

import asyncio
import os

import click
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
)
from dotenv import load_dotenv

from .semantic_catalog import load_catalog
from .tools import build_semantic_tools

SYSTEM_PROMPT = """\
You are a conversational *analytics* assistant over a MusicBrainz music \
metadata database, grounded entirely in an Ossie semantic model — not raw \
SQL tables. You answer analytical questions (counts, averages, rates, \
breakdowns, trends over time/genre/region) — you are not a catalog browser \
or a lookup service for exhaustive raw listings.

Workflow for every question:
1. Call list_semantic_models ONCE per conversation (it's the same every \
   time — never call it again in the same session). It returns full detail \
   for the curated datasets/metrics that cover real analytics; the rest of \
   the schema (editorial/moderation/import-staging tables) is intentionally \
   not listed, since it's rarely relevant here — if you have a specific, \
   well-founded reason to need one, call describe_model directly on its \
   real MusicBrainz table name.
2. If you're unsure which fields/relationships a dataset has, call \
   describe_model on it before guessing a field name.
3. Call run_semantic_query with an anchor dataset, the metric(s) that answer \
   the question, and any dimensions/filters needed. Joins across datasets \
   are resolved automatically — you never name a physical table or write SQL.
   Keep requests narrow: ask for the aggregated numbers that answer the \
   question (top N, a breakdown by a dimension, a trend by year), not wide \
   raw-row dumps — a query requesting many rows/columns at once can itself \
   exceed the tool output limit and cost an extra round trip to recover \
   from. If you only need to confirm an entity's identity (e.g. resolving \
   an artist name), request just id/name, not every field.
4. Name lookups often miss on the first try: MusicBrainz canonical names \
   can be unusual (unicode punctuation, alternate stylization). Use ILIKE \
   with wildcards before assuming something doesn't exist, and check the \
   relevant *_alias dataset (e.g. artist_alias) if a direct name search \
   comes up empty — the commonly-known name is often stored as an alias, \
   not the canonical name.
5. Answer in plain language. State the actual numbers returned, and briefly \
   note which metric(s)/dataset(s) you used so the answer is auditable. If a \
   tool call fails, read the error, adjust (e.g. pick a different metric/\
   field name from describe_model), and retry — don't guess blindly.

If a question can't be answered with the datasets/metrics available, say so \
explicitly rather than inventing numbers.
"""


def build_options(model: str) -> ClaudeAgentOptions:
    catalog = load_catalog()
    dsn = (
        f"host={os.environ['PGHOST']} port={os.environ['PGPORT']} dbname={os.environ['PGDATABASE']} "
        f"user={os.environ['MB_RO_USER']} password={os.environ['MB_RO_PASSWORD']}"
    )
    server = create_sdk_mcp_server(
        name="musicbrainz",
        tools=build_semantic_tools(catalog, dsn),
    )
    return ClaudeAgentOptions(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"musicbrainz": server},
        tools=[],  # no built-in Claude Code tools (Bash, Read, Write, ...) — semantic tools only
        allowed_tools=[
            "mcp__musicbrainz__list_semantic_models",
            "mcp__musicbrainz__describe_model",
            "mcp__musicbrainz__run_semantic_query",
        ],
        # Deliberately NOT setting permission_mode="bypassPermissions" here:
        # tried it while chasing a hang that turned out to be unrelated (a
        # nested-Claude-session artifact in the debugging environment, not
        # this code), and it caused a real hang in normal terminal use —
        # bypassPermissions mode likely has its own one-time interactive
        # confirmation that a non-interactive SDK session can't answer.
        # allowed_tools above already scopes the agent to 3 narrow read-only
        # tools, which is the actual safety boundary; the default permission
        # mode has never hung in real terminal testing.
    )


def _print_message(msg) -> None:
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(f"\nassistant: {block.text}")
            elif isinstance(block, ToolUseBlock):
                print(f"  [using {block.name}: {block.input}]")
    elif isinstance(msg, ResultMessage) and msg.total_cost_usd:
        print(f"  (cost: ${msg.total_cost_usd:.4f})")


async def run_chat(model: str) -> None:
    load_dotenv()
    options = build_options(model)

    print(f"MusicBrainz conversational analytics (model: {model}) — type a question, or 'exit' to quit.\n")
    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                question = input("you: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break

            await client.query(question)
            async for msg in client.receive_response():
                _print_message(msg)


@click.command()
@click.option(
    "--model",
    default="claude-sonnet-5",
    show_default=True,
    help="Model to run the agent on, e.g. claude-sonnet-5, claude-haiku-4-5-20251001, claude-opus-5.",
)
def main(model: str) -> None:
    asyncio.run(run_chat(model))


if __name__ == "__main__":
    main()
