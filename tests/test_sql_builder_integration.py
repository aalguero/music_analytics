"""Executes sql_builder.py's output against a real Postgres schema — proves
the generated SQL actually runs and returns correct joined/aggregated/
filtered results, not just that it's shaped right as a string (that's what
test_sql_builder.py checks). Uses a small synthetic catalog (not the real
semantic/core/ content, which points at a database named "musicbrainz" —
see the module docstring in sql_builder.py on why the source prefix has to
match the connected database name) mirroring the shape of the real
release/artist/artist_credit/artist_credit_name/release_country tables.
"""

import psycopg
import pytest
import sqlalchemy as sa

from agent.semantic_catalog import SemanticCatalog
from agent.sql_builder import Filter, build_query
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

TEST_DB_URL = "postgresql+psycopg://andre@/music_analytics_test"
SCHEMA = "mb_sql_builder_test"
SOURCE_PREFIX = f"music_analytics_test.{SCHEMA}"  # leading part must equal current_database()

DDL = f"""
CREATE SCHEMA {SCHEMA};

CREATE TABLE {SCHEMA}.artist (
    id serial PRIMARY KEY,
    name varchar NOT NULL
);

CREATE TABLE {SCHEMA}.artist_credit (
    id serial PRIMARY KEY,
    name varchar NOT NULL
);

CREATE TABLE {SCHEMA}.artist_credit_name (
    artist_credit integer NOT NULL,
    "position" smallint NOT NULL,
    artist integer NOT NULL
);

CREATE TABLE {SCHEMA}.release (
    id serial PRIMARY KEY,
    name varchar NOT NULL,
    artist_credit integer NOT NULL,
    status integer
);

CREATE TABLE {SCHEMA}.release_country (
    release integer NOT NULL,
    date_year integer
);

INSERT INTO {SCHEMA}.artist (id, name) VALUES (1, 'Alice'), (2, 'Bob');
INSERT INTO {SCHEMA}.artist_credit (id, name) VALUES (1, 'Alice'), (2, 'Bob');
INSERT INTO {SCHEMA}.artist_credit_name (artist_credit, "position", artist) VALUES (1, 0, 1), (2, 0, 2);
INSERT INTO {SCHEMA}.release (id, name, artist_credit, status) VALUES
    (1, 'Album A', 1, 1),
    (2, 'Album B', 1, 1),
    (3, 'Album C', 2, 1);
INSERT INTO {SCHEMA}.release_country (release, date_year) VALUES
    (1, 1990),
    (2, 1995),
    (3, 2000);
"""


def _field(name, expr, datatype=OSIDataType.STRING):
    return OSIField(
        name=name,
        expression=OSIExpression(dialects=[OSIDialectExpression(dialect=OSIDialect.ANSI_SQL, expression=expr)]),
        datatype=datatype,
    )


@pytest.fixture(scope="module")
def engine():
    eng = sa.create_engine(TEST_DB_URL)
    with eng.begin() as conn:
        conn.execute(sa.text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        for stmt in DDL.strip().split(";"):
            if stmt.strip():
                conn.execute(sa.text(stmt))
    yield eng
    with eng.begin() as conn:
        conn.execute(sa.text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    eng.dispose()


@pytest.fixture(scope="module")
def catalog():
    """Mirrors the shape (names, relationships) of the real
    artist/artist_credit/artist_credit_name/release/release_country
    datasets in semantic/core/, but pointed at the sandbox schema."""
    cat = SemanticCatalog()

    cat.datasets["artist"] = OSIDataset(
        name="artist",
        source=f"{SOURCE_PREFIX}.artist",
        primary_key=["id"],
        fields=[_field("id", "id", OSIDataType.INTEGER), _field("name", "name")],
    )
    cat.datasets["artist_credit"] = OSIDataset(
        name="artist_credit",
        source=f"{SOURCE_PREFIX}.artist_credit",
        primary_key=["id"],
        fields=[_field("id", "id", OSIDataType.INTEGER)],
    )
    cat.datasets["artist_credit_name"] = OSIDataset(
        name="artist_credit_name",
        source=f"{SOURCE_PREFIX}.artist_credit_name",
        primary_key=["artist_credit", "position"],
        fields=[
            _field("artist_credit", "artist_credit", OSIDataType.INTEGER),
            _field("artist", "artist", OSIDataType.INTEGER),
        ],
    )
    cat.datasets["release"] = OSIDataset(
        name="release",
        source=f"{SOURCE_PREFIX}.release",
        primary_key=["id"],
        fields=[
            _field("id", "id", OSIDataType.INTEGER),
            _field("name", "name"),
            _field("artist_credit", "artist_credit", OSIDataType.INTEGER),
            _field("status", "status", OSIDataType.INTEGER),
        ],
    )
    cat.datasets["release_country"] = OSIDataset(
        name="release_country",
        source=f"{SOURCE_PREFIX}.release_country",
        primary_key=["release"],
        fields=[
            _field("release", "release", OSIDataType.INTEGER),
            _field("date_year", "date_year", OSIDataType.INTEGER),
        ],
    )

    def rel(name, frm, to, from_cols, to_cols):
        return OSIRelationship(name=name, from_dataset=frm, to=to, from_columns=from_cols, to_columns=to_cols)

    cat.relationships["release_to_artist_credit"] = rel(
        "release_to_artist_credit", "release", "artist_credit", ["artist_credit"], ["id"]
    )
    cat.relationships["acn_to_artist_credit"] = rel(
        "acn_to_artist_credit", "artist_credit_name", "artist_credit", ["artist_credit"], ["id"]
    )
    cat.relationships["acn_to_artist"] = rel("acn_to_artist", "artist_credit_name", "artist", ["artist"], ["id"])
    cat.relationships["release_country_to_release"] = rel(
        "release_country_to_release", "release_country", "release", ["release"], ["id"]
    )

    cat.metrics["release_count"] = OSIMetric(
        name="release_count",
        expression=OSIExpression(
            dialects=[OSIDialectExpression(dialect=OSIDialect.ANSI_SQL, expression="COUNT(DISTINCT release.id)")]
        ),
        datatype=OSIDataType.INTEGER,
    )

    return cat


def _run(engine, query):
    with psycopg.connect("dbname=music_analytics_test user=andre", row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query.sql, query.params)
            return cur.fetchall()


def test_multi_hop_join_executes_and_groups_correctly(engine, catalog):
    query = build_query(catalog, dataset="release", metrics=["release_count"], dimensions=["artist.name"])
    rows = _run(engine, query)

    by_artist = {row["artist.name"]: row["release_count"] for row in rows}
    assert by_artist == {"Alice": 2, "Bob": 1}


def test_filter_via_joined_dataset_executes_correctly(engine, catalog):
    query = build_query(
        catalog,
        dataset="release",
        metrics=["release_count"],
        dimensions=["artist.name"],
        filters=[Filter(field="release_country.date_year", op=">=", value=1993)],
    )
    rows = _run(engine, query)

    by_artist = {row["artist.name"]: row["release_count"] for row in rows}
    # Only releases 2 (1995, Alice) and 3 (2000, Bob) pass the >= 1993 filter.
    assert by_artist == {"Alice": 1, "Bob": 1}


def test_in_filter_executes_correctly(engine, catalog):
    query = build_query(
        catalog,
        dataset="release",
        metrics=["release_count"],
        filters=[Filter(field="release.id", op="IN", value=[1, 2])],
    )
    rows = _run(engine, query)
    assert rows == [{"release_count": 2}]


def test_dimension_only_select_distinct_executes(engine, catalog):
    query = build_query(catalog, dataset="artist", dimensions=["artist.name"])
    rows = _run(engine, query)
    names = {row["artist.name"] for row in rows}
    assert names == {"Alice", "Bob"}
