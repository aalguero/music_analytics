"""Tests agent/tools.py by calling each SdkMcpTool's .handler(args) directly
— this is the actual async function claude-agent-sdk would invoke, so it
exercises real behavior without needing a live Claude session/API key.
run_semantic_query is tested against a real sandbox Postgres schema, same
pattern as test_sql_builder_integration.py.
"""

import json

import psycopg
import pytest

from agent.semantic_catalog import SemanticCatalog
from agent.tools import build_semantic_tools
from ossie.models import (
    OSIDataset,
    OSIDataType,
    OSIDialect,
    OSIDialectExpression,
    OSIExpression,
    OSIField,
    OSIMetric,
    OSIRelationship,
)

DSN = "dbname=music_analytics_test user=andre"
SCHEMA = "mb_tools_test"
SOURCE_PREFIX = f"music_analytics_test.{SCHEMA}"

DDL = f"""
CREATE SCHEMA {SCHEMA};

CREATE TABLE {SCHEMA}.genre (
    id serial PRIMARY KEY,
    name varchar NOT NULL
);

INSERT INTO {SCHEMA}.genre (id, name) VALUES (1, 'jazz'), (2, 'synthpop'), (3, 'ambient');
"""


def _field(name, expr, datatype=OSIDataType.STRING, description=None):
    return OSIField(
        name=name,
        expression=OSIExpression(dialects=[OSIDialectExpression(dialect=OSIDialect.ANSI_SQL, expression=expr)]),
        datatype=datatype,
        description=description,
    )


@pytest.fixture(scope="module", autouse=True)
def sandbox_schema():
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
            cur.execute(DDL)
    yield
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")


@pytest.fixture
def catalog():
    cat = SemanticCatalog()
    cat.datasets["genre"] = OSIDataset(
        name="genre",
        source=f"{SOURCE_PREFIX}.genre",
        primary_key=["id"],
        description="A curated music genre.",
        fields=[
            _field("id", "id", OSIDataType.INTEGER),
            _field("name", "name", description="Genre name"),
        ],
    )
    cat.datasets["other"] = OSIDataset(
        name="other",
        source=f"{SOURCE_PREFIX}.other",
        primary_key=["id"],
        fields=[_field("id", "id", OSIDataType.INTEGER)],
    )
    cat.relationships["other_to_genre"] = OSIRelationship(
        name="other_to_genre", from_dataset="other", to="genre", from_columns=["id"], to_columns=["id"]
    )
    cat.metrics["genre_count"] = OSIMetric(
        name="genre_count",
        expression=OSIExpression(
            dialects=[OSIDialectExpression(dialect=OSIDialect.ANSI_SQL, expression="COUNT(DISTINCT genre.id)")]
        ),
        datatype=OSIDataType.INTEGER,
        description="Total genres",
    )
    return cat


@pytest.fixture
def tools(catalog):
    return {t.name: t for t in build_semantic_tools(catalog, DSN)}


def _payload(result):
    assert not result.get("is_error"), result
    return json.loads(result["content"][0]["text"])


async def test_list_semantic_models(tools):
    result = await tools["list_semantic_models"].handler({})
    payload = _payload(result)
    # The synthetic catalog fixture doesn't go through load_catalog(), so
    # nothing has a tracked origin — dataset_origin()/metric_origin() both
    # default to "core", landing everything in the curated_* lists.
    curated_names = {d["dataset"] for d in payload["curated_datasets"]}
    assert curated_names == {"genre", "other"}
    assert payload["other_uncurated_dataset_count"] == 0
    curated_metric_names = {m["metric"] for m in payload["curated_metrics"]}
    assert curated_metric_names == {"genre_count"}
    assert payload["other_uncurated_metric_count"] == 0


async def test_list_semantic_models_splits_generated_to_counts_only(catalog):
    # Mirrors what load_catalog() actually records for a mechanically
    # generated table — full detail (or even just names) would be
    # redundant/bloated at scale (this is what blew the tool-output token
    # limit against the real 355-table generated catalog before this split
    # existed).
    catalog._dataset_origin["other"] = "generated"
    catalog._metric_origin["genre_count"] = "generated"
    tools = {t.name: t for t in build_semantic_tools(catalog, DSN)}

    result = await tools["list_semantic_models"].handler({})
    payload = _payload(result)

    assert {d["dataset"] for d in payload["curated_datasets"]} == {"genre"}
    assert payload["other_uncurated_dataset_count"] == 1
    assert payload["curated_metrics"] == []
    assert payload["other_uncurated_metric_count"] == 1


async def test_describe_model(tools):
    result = await tools["describe_model"].handler({"dataset": "genre"})
    payload = _payload(result)
    assert payload["dataset"] == "genre"
    field_names = {f["name"] for f in payload["fields"]}
    assert field_names == {"id", "name"}
    assert payload["relationships"][0]["from"] == "other"
    assert payload["relationships"][0]["to"] == "genre"


async def test_describe_model_unknown_dataset_returns_error_not_exception(tools):
    result = await tools["describe_model"].handler({"dataset": "does_not_exist"})
    assert result["is_error"] is True
    assert "Unknown dataset" in result["content"][0]["text"]


async def test_run_semantic_query_executes_against_real_db(tools):
    result = await tools["run_semantic_query"].handler({"dataset": "genre", "metrics": ["genre_count"]})
    payload = _payload(result)
    assert payload["row_count"] == 1
    assert payload["rows"][0]["genre_count"] == 3
    assert "sql" in payload


async def test_run_semantic_query_dimension_only(tools):
    result = await tools["run_semantic_query"].handler({"dataset": "genre", "dimensions": ["genre.name"]})
    payload = _payload(result)
    names = {row["genre.name"] for row in payload["rows"]}
    assert names == {"jazz", "synthpop", "ambient"}


async def test_run_semantic_query_bad_metric_returns_error_not_exception(tools):
    result = await tools["run_semantic_query"].handler({"dataset": "genre", "metrics": ["not_a_real_metric"]})
    assert result["is_error"] is True
    assert "Could not build query" in result["content"][0]["text"]


async def test_run_semantic_query_filter(tools):
    result = await tools["run_semantic_query"].handler(
        {
            "dataset": "genre",
            "dimensions": ["genre.name"],
            "filters": [{"field": "genre.name", "op": "!=", "value": "jazz"}],
        }
    )
    payload = _payload(result)
    names = {row["genre.name"] for row in payload["rows"]}
    assert names == {"synthpop", "ambient"}
