"""The three tools the agent gets, wired up as an in-process claude-agent-sdk
MCP server. These are the *only* way the agent touches data — it never sees
raw SQL table names, and it never gets a general-purpose SQL execution tool.

- list_semantic_models: catalog overview (grounding — "what exists").
- describe_model: full detail on one dataset (fields, relationships).
- run_semantic_query: the only tool that touches Postgres, and only via
  sql_builder.py (parameterized, catalog-name-only SQL) against the
  read-only `mb_analytics_ro` role — see create_readonly_role.sql.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, NotRequired, TypedDict

import psycopg
from claude_agent_sdk import SdkMcpTool, tool

from .semantic_catalog import CatalogError, SemanticCatalog, ai_context_dict
from .sql_builder import Filter, SqlBuildError, build_query


class QueryFilterInput(TypedDict):
    field: Annotated[str, "Qualified as 'dataset.field', e.g. 'release_country.date_year'"]
    op: Annotated[
        str,
        "One of: =, !=, >, >=, <, <=, IN, NOT IN, ILIKE, NOT ILIKE. "
        "Use ILIKE with % wildcards for case-insensitive/partial text matches "
        "(e.g. value='%dmc%') — exact name matches on real data often fail due to "
        "unusual canonical spellings/punctuation (curly hyphens, periods, etc.); "
        "also consider searching *_alias datasets (see artist_alias) since MusicBrainz "
        "often stores the commonly-known name as an alias, not the canonical name.",
    ]
    value: Annotated[Any, "Scalar for most ops; a list for IN / NOT IN"]


class RunSemanticQueryInput(TypedDict):
    dataset: Annotated[str, "The base/anchor dataset name to query from, e.g. 'release'"]
    metrics: NotRequired[Annotated[list[str], "Metric names from describe_model/list_semantic_models"]]
    dimensions: NotRequired[Annotated[list[str], "Qualified 'dataset.field' names to group/select by"]]
    filters: NotRequired[list[QueryFilterInput]]
    limit: NotRequired[Annotated[int, "Max rows to return (default 50, hard cap 500)"]]


def _text_result(payload: dict) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}]}


def _error_result(message: str) -> dict:
    return {"content": [{"type": "text", "text": message}], "is_error": True}


def build_semantic_tools(catalog: SemanticCatalog, dsn: str, *, default_limit: int = 50) -> list[SdkMcpTool]:
    @tool(
        "list_semantic_models",
        "List available datasets and metrics for analytical queries. Returns full detail "
        "(description, synonyms, when to use) for the curated datasets/metrics — these cover "
        "the real analytics: artists, releases, recordings, genres, labels, works, and their "
        "counts/averages/rates. The other ~350 tables in the raw MusicBrainz schema (editorial "
        "history, moderation, raw import staging, etc.) are auto-generated and NOT listed here "
        "by name to keep this compact — they're rarely relevant to analytical questions. If you "
        "have a specific reason to need one and know or can guess its real MusicBrainz table "
        "name, call describe_model on it directly; an unknown-name error will suggest close "
        "matches. Call this once per conversation, not per question — it doesn't change.",
        {},
    )
    async def list_semantic_models(_args: dict) -> dict:
        curated_datasets = [entry for entry in catalog.summary() if catalog.dataset_origin(entry["dataset"]) == "core"]
        curated_metrics = [entry for entry in catalog.metrics_summary() if catalog.metric_origin(entry["metric"]) == "core"]
        other_dataset_count = sum(1 for name in catalog.datasets if catalog.dataset_origin(name) != "core")
        other_metric_count = sum(1 for name in catalog.metrics if catalog.metric_origin(name) != "core")
        return _text_result(
            {
                "curated_datasets": curated_datasets,
                "curated_metrics": curated_metrics,
                "other_uncurated_dataset_count": other_dataset_count,
                "other_uncurated_metric_count": other_metric_count,
            }
        )

    @tool(
        "describe_model",
        "Get full detail on one dataset: its fields (with datatypes and synonyms) and its "
        "relationships to other datasets. Call this before run_semantic_query if you're not "
        "sure which field/relationship names are available on a dataset.",
        {"dataset": str},
    )
    async def describe_model(args: dict) -> dict:
        dataset_name = args["dataset"]
        try:
            dataset = catalog.dataset(dataset_name)
        except CatalogError as e:
            return _error_result(str(e))

        fields = [
            {
                "name": f.name,
                "datatype": f.datatype,
                "description": f.description,
                "ai_context": ai_context_dict(f.ai_context),
            }
            for f in dataset.fields or []
        ]
        relationships = [
            {"name": r.name, "from": r.from_dataset, "to": r.to, "from_columns": r.from_columns, "to_columns": r.to_columns}
            for r in catalog.relationships_touching(dataset_name)
        ]
        return _text_result(
            {
                "dataset": dataset_name,
                "description": dataset.description,
                "fields": fields,
                "relationships": relationships,
            }
        )

    @tool(
        "run_semantic_query",
        "Run an analytical query defined purely in terms of the semantic model: an anchor "
        "dataset, metric names, dimension names ('dataset.field'), and optional filters. "
        "Joins are resolved automatically from the catalog's declared relationships — you "
        "never write SQL or name a physical table. Read-only; capped at 500 rows.",
        RunSemanticQueryInput,
    )
    async def run_semantic_query(args: dict) -> dict:
        filters = [Filter(field=f["field"], op=f["op"], value=f["value"]) for f in args.get("filters", [])]
        try:
            query = build_query(
                catalog,
                dataset=args["dataset"],
                metrics=args.get("metrics"),
                dimensions=args.get("dimensions"),
                filters=filters,
                limit=args.get("limit", default_limit),
            )
        except (SqlBuildError, CatalogError) as e:
            return _error_result(f"Could not build query: {e}")

        try:
            with psycopg.connect(dsn, row_factory=psycopg.rows.dict_row) as conn:
                conn.read_only = True  # defense in depth on top of the mb_analytics_ro role's grants
                with conn.cursor() as cur:
                    cur.execute(query.sql, query.params)
                    rows = cur.fetchall()
        except psycopg.Error as e:
            return _error_result(f"Query failed: {e}")

        return _text_result({"sql": query.sql, "row_count": len(rows), "rows": rows})

    return [list_semantic_models, describe_model, run_semantic_query]
