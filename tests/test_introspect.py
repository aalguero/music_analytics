"""Integration test for semantic/generator/introspect.py against a real
Postgres schema (not a mock) — set up and torn down entirely within this
test using the `music_analytics_test` database.

Uses a handful of tables that mirror real MusicBrainz DDL fragments closely
enough to exercise the interesting cases: a table already covered by
semantic/core/ (`area`, should be skipped), plain lookup tables that should
get generated, a self-referential foreign key, and a column type sweep
(UUID, boolean, numeric, array, timestamptz, date, time) to check
_map_datatype against what Postgres actually reflects back.
"""

import os

import pytest
import sqlalchemy as sa
import yaml
from ossie.models import OSIDocument

from semantic.generator.introspect import generate_semantic_layer

TEST_DB_URL = "postgresql+psycopg://andre@/music_analytics_test"
SCHEMA = "mb_introspect_test"

DDL = f"""
CREATE SCHEMA {SCHEMA};

CREATE TABLE {SCHEMA}.area (
    id serial PRIMARY KEY,
    gid uuid NOT NULL,
    name varchar NOT NULL
);

CREATE TABLE {SCHEMA}.gender (
    id serial PRIMARY KEY,
    name varchar NOT NULL
);

CREATE TABLE {SCHEMA}.artist_type (
    id serial PRIMARY KEY,
    gid uuid NOT NULL,
    name varchar NOT NULL,
    parent integer REFERENCES {SCHEMA}.artist_type(id)
);

CREATE TABLE {SCHEMA}.release_extra (
    id serial PRIMARY KEY,
    is_va boolean NOT NULL,
    catalog_numbers text[],
    rating numeric(5,2),
    created_at timestamptz,
    event_date date,
    event_time time,
    area_ref integer REFERENCES {SCHEMA}.area(id)
);
"""


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


@pytest.fixture()
def output_dir(tmp_path):
    return tmp_path / "generated"


@pytest.fixture()
def results(engine, output_dir):
    return generate_semantic_layer(
        engine,
        schema=SCHEMA,
        source_prefix=f"music_analytics_test.{SCHEMA}",
        core_dir="semantic/core",
        output_dir=output_dir,
    )


def _load(output_dir, table):
    with open(output_dir / f"{table}.ossie.yaml") as f:
        # Skip the auto-generated header comment lines before parsing.
        return yaml.safe_load(f)


def test_core_covered_table_is_skipped(results, output_dir):
    generated_tables = {r.table for r in results}
    assert "area" not in generated_tables
    assert not (output_dir / "area.ossie.yaml").exists()


def test_uncovered_tables_are_generated(results):
    generated_tables = {r.table for r in results}
    assert generated_tables == {"gender", "artist_type", "release_extra"}


def test_generated_file_is_valid_ossie(results, output_dir):
    raw = _load(output_dir, "gender")
    doc = OSIDocument.model_validate(raw)
    dataset = doc.semantic_model[0].datasets[0]
    assert dataset.name == "gender"
    assert dataset.source == f"music_analytics_test.{SCHEMA}.gender"
    assert dataset.primary_key == ["id"]
    field_names = {f.name for f in dataset.fields}
    assert field_names == {"id", "name"}


def test_self_referential_fk_becomes_relationship(results, output_dir):
    raw = _load(output_dir, "artist_type")
    doc = OSIDocument.model_validate(raw)
    model = doc.semantic_model[0]
    rels = model.relationships
    assert rels and len(rels) == 1
    rel = rels[0]
    assert rel.from_dataset == "artist_type"
    assert rel.to == "artist_type"
    assert rel.from_columns == ["parent"]
    assert rel.to_columns == ["id"]


def test_fk_to_core_covered_table_still_generated(results, output_dir):
    raw = _load(output_dir, "release_extra")
    doc = OSIDocument.model_validate(raw)
    rels = doc.semantic_model[0].relationships
    assert rels and len(rels) == 1
    rel = rels[0]
    assert rel.to == "area"  # core-covered, but the relationship is still emitted
    assert rel.from_columns == ["area_ref"]


def test_datatype_mapping(results, output_dir):
    raw = _load(output_dir, "release_extra")
    doc = OSIDocument.model_validate(raw)
    fields = {f.name: f for f in doc.semantic_model[0].datasets[0].fields}

    assert fields["is_va"].datatype == "Boolean"
    assert fields["rating"].datatype == "Decimal"
    assert fields["created_at"].datatype == "DateTimeTz"
    assert fields["event_date"].datatype == "Date"
    assert fields["event_time"].datatype == "Time"
    assert fields["area_ref"].datatype == "Integer"
    # Array types are outside the portable vocabulary — datatype omitted.
    assert fields["catalog_numbers"].datatype is None


def test_gid_uuid_maps_to_string(results, output_dir):
    raw = _load(output_dir, "artist_type")
    doc = OSIDocument.model_validate(raw)
    fields = {f.name: f for f in doc.semantic_model[0].datasets[0].fields}
    assert fields["gid"].datatype == "String"


def test_count_metric_generated(results, output_dir):
    raw = _load(output_dir, "gender")
    doc = OSIDocument.model_validate(raw)
    metrics = doc.semantic_model[0].metrics
    assert metrics and metrics[0].name == "gender_count"
    assert "COUNT(DISTINCT gender.id)" in metrics[0].expression.dialects[0].expression
