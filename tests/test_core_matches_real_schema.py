"""Cross-checks every dataset/field declared in semantic/core/*.ossie.yaml
against a schema built from MusicBrainz's actual column names (transcribed
from admin/sql/CreateTables.sql in metabrainz/musicbrainz-server). Column
*types* here are simplified/approximate — this test only protects against
the failure mode that actually matters for a hand-authored semantic layer:
a field referencing a column name that doesn't really exist, e.g. a typo. It
intentionally does not depend on `dataset.source`'s db/schema prefix (see
sql_builder.py's docstring on why that prefix must equal the connected
database name) — it only compares bare table/column names.

If this test fails after a MusicBrainz schema migration, the fix belongs in
semantic/core/*.ossie.yaml, not here — this file exists to catch drift *from
the model*, so update the DDL below only when re-verifying against a fresh
CreateTables.sql, not to make a failing test pass.
"""

import pytest
import sqlalchemy as sa

from agent.semantic_catalog import load_catalog

TEST_DB_URL = "postgresql+psycopg://andre@/music_analytics_test"
SCHEMA = "mb_real_columns_test"

# Column names transcribed verbatim from metabrainz/musicbrainz-server's
# admin/sql/CreateTables.sql (types simplified — see module docstring).
DDL = f"""
CREATE SCHEMA {SCHEMA};

CREATE TABLE {SCHEMA}.artist (
    id serial, gid uuid, name varchar, sort_name varchar,
    begin_date_year smallint, begin_date_month smallint, begin_date_day smallint,
    end_date_year smallint, end_date_month smallint, end_date_day smallint,
    type integer, area integer, gender integer, comment varchar,
    edits_pending integer, last_updated timestamptz, ended boolean,
    begin_area integer, end_area integer
);

CREATE TABLE {SCHEMA}.artist_credit (
    id serial, name varchar, artist_count smallint, ref_count integer,
    created timestamptz, edits_pending integer, gid uuid
);

CREATE TABLE {SCHEMA}.artist_credit_name (
    artist_credit integer, "position" smallint, artist integer,
    name varchar, join_phrase text
);

CREATE TABLE {SCHEMA}.release (
    id serial, gid uuid, name varchar, artist_credit integer, release_group integer,
    status integer, packaging integer, language integer, script integer,
    barcode varchar, comment varchar, edits_pending integer, quality smallint,
    last_updated timestamptz
);

CREATE TABLE {SCHEMA}.release_country (
    release integer, country integer, date_year smallint, date_month smallint, date_day smallint
);

CREATE TABLE {SCHEMA}.release_unknown_country (
    release integer, date_year smallint, date_month smallint, date_day smallint
);

CREATE TABLE {SCHEMA}.release_label (
    id serial, release integer, label integer, catalog_number varchar, last_updated timestamptz
);

CREATE TABLE {SCHEMA}.release_group (
    id serial, gid uuid, name varchar, artist_credit integer, type integer,
    comment varchar, edits_pending integer, last_updated timestamptz
);

CREATE TABLE {SCHEMA}.recording (
    id serial, gid uuid, name varchar, artist_credit integer, length integer,
    comment varchar, edits_pending integer, last_updated timestamptz, video boolean
);

CREATE TABLE {SCHEMA}.work (
    id serial, gid uuid, name varchar, type integer, comment varchar,
    edits_pending integer, last_updated timestamptz
);

CREATE TABLE {SCHEMA}.l_artist_work (
    id serial, link integer, entity0 integer, entity1 integer,
    edits_pending integer, last_updated timestamptz, link_order integer,
    entity0_credit text, entity1_credit text
);

CREATE TABLE {SCHEMA}.l_recording_work (
    id serial, link integer, entity0 integer, entity1 integer,
    edits_pending integer, last_updated timestamptz, link_order integer,
    entity0_credit text, entity1_credit text
);

CREATE TABLE {SCHEMA}.label (
    id serial, gid uuid, name varchar,
    begin_date_year smallint, begin_date_month smallint, begin_date_day smallint,
    end_date_year smallint, end_date_month smallint, end_date_day smallint,
    label_code integer, type integer, area integer, comment varchar,
    edits_pending integer, last_updated timestamptz, ended boolean
);

CREATE TABLE {SCHEMA}.area (
    id serial, gid uuid, name varchar, type integer, edits_pending integer,
    last_updated timestamptz, begin_date_year smallint, begin_date_month smallint,
    begin_date_day smallint, end_date_year smallint, end_date_month smallint,
    end_date_day smallint, ended boolean, comment varchar
);

CREATE TABLE {SCHEMA}.genre (
    id serial, gid uuid, name varchar, comment varchar,
    edits_pending integer, last_updated timestamptz
);

CREATE TABLE {SCHEMA}.l_artist_genre (
    id serial, link integer, entity0 integer, entity1 integer,
    edits_pending integer, last_updated timestamptz, link_order integer,
    entity0_credit text, entity1_credit text
);

CREATE TABLE {SCHEMA}.l_genre_recording (
    id serial, link integer, entity0 integer, entity1 integer,
    edits_pending integer, last_updated timestamptz, link_order integer,
    entity0_credit text, entity1_credit text
);

CREATE TABLE {SCHEMA}.l_genre_release (
    id serial, link integer, entity0 integer, entity1 integer,
    edits_pending integer, last_updated timestamptz, link_order integer,
    entity0_credit text, entity1_credit text
);

CREATE TABLE {SCHEMA}.l_genre_release_group (
    id serial, link integer, entity0 integer, entity1 integer,
    edits_pending integer, last_updated timestamptz, link_order integer,
    entity0_credit text, entity1_credit text
);
"""


@pytest.fixture(scope="module")
def real_columns():
    engine = sa.create_engine(TEST_DB_URL)
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        for stmt in DDL.strip().split(";"):
            if stmt.strip():
                conn.execute(sa.text(stmt))
    inspector = sa.inspect(engine)
    columns = {
        table: {c["name"] for c in inspector.get_columns(table, schema=SCHEMA)}
        for table in inspector.get_table_names(schema=SCHEMA)
    }
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    engine.dispose()
    return columns


def test_every_core_field_matches_a_real_column(real_columns):
    catalog = load_catalog()
    mismatches = []
    checked = 0

    for name, dataset in catalog.datasets.items():
        table = dataset.source.rsplit(".", 1)[-1]
        if table not in real_columns:
            continue  # not one of the tables verified in this test's DDL — skip, don't fail
        for f in dataset.fields or []:
            expr = f.expression.dialects[0].expression.strip('"')
            checked += 1
            if expr not in real_columns[table] and not expr.startswith("("):
                mismatches.append(f"{name}.{f.name} -> column '{expr}' not found on real table '{table}'")

    assert checked > 50, f"only checked {checked} fields — DDL coverage looks too small, test isn't exercising much"
    assert not mismatches, "\n" + "\n".join(mismatches)
