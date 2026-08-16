"""Unit tests for sql_builder.py. Two kinds of coverage:

1. Against the *real* semantic/core/ catalog — multi-hop joins, filters, and
   dimension-only queries, checking the actual join chains this project's
   relationships produce (e.g. release -> artist_credit -> artist_credit_name
   -> artist).
2. Against small synthetic catalogs — error paths (unknown names, bad
   filter ops, no join path) that shouldn't depend on real content changing.
"""

import pytest

from agent.semantic_catalog import SemanticCatalog, load_catalog
from agent.sql_builder import Filter, SqlBuildError, build_query
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


@pytest.fixture(scope="module")
def real_catalog():
    return load_catalog()


def _field(name, expr, datatype=OSIDataType.STRING):
    return OSIField(
        name=name,
        expression=OSIExpression(dialects=[OSIDialectExpression(dialect=OSIDialect.ANSI_SQL, expression=expr)]),
        datatype=datatype,
    )


def _dataset(name, source, fields):
    return OSIDataset(name=name, source=source, primary_key=["id"], fields=fields)


@pytest.fixture
def disconnected_catalog():
    """Two datasets with no relationship between them at all."""
    catalog = SemanticCatalog()
    catalog.datasets["island_a"] = _dataset("island_a", "db.s.island_a", [_field("id", "id", OSIDataType.INTEGER)])
    catalog.datasets["island_b"] = _dataset("island_b", "db.s.island_b", [_field("id", "id", OSIDataType.INTEGER)])
    return catalog


def _rel(name, frm, to):
    return OSIRelationship(name=name, from_dataset=frm, to=to, from_columns=["id"], to_columns=["id"])


@pytest.fixture
def hub_shortcut_catalog():
    """Reproduces the real bug found in Stage 5 validation: a longer,
    hand-curated (core) path between fact_a and fact_b coexists with a
    shorter, mechanically-generated shortcut through an unrelated `hub`
    dataset (like release_country and artist both happening to reference
    `area` for unrelated reasons). The shorter path must lose."""
    catalog = SemanticCatalog()
    for name in ["fact_a", "bridge1", "bridge2", "fact_b", "hub"]:
        catalog.datasets[name] = _dataset(name, f"db.s.{name}", [_field("id", "id", OSIDataType.INTEGER)])

    # Real 3-hop path, curated by hand.
    for rel in [_rel("a_to_b1", "fact_a", "bridge1"), _rel("b1_to_b2", "bridge1", "bridge2"), _rel("b2_to_b", "bridge2", "fact_b")]:
        catalog.relationships[rel.name] = rel
        catalog._relationship_origin[rel.name] = "core"

    # Shorter but semantically meaningless 2-hop shortcut, mechanically generated.
    for rel in [_rel("a_to_hub", "fact_a", "hub"), _rel("hub_to_b", "hub", "fact_b")]:
        catalog.relationships[rel.name] = rel
        catalog._relationship_origin[rel.name] = "generated"

    catalog.metrics["fact_b_count"] = OSIMetric(
        name="fact_b_count",
        expression=OSIExpression(
            dialects=[OSIDialectExpression(dialect=OSIDialect.ANSI_SQL, expression="COUNT(DISTINCT fact_b.id)")]
        ),
    )
    return catalog


def test_core_path_preferred_over_shorter_generated_shortcut(hub_shortcut_catalog):
    query = build_query(hub_shortcut_catalog, dataset="fact_a", metrics=["fact_b_count"])

    assert "hub" not in query.datasets_used
    assert query.datasets_used == ["bridge1", "bridge2", "fact_a", "fact_b"]


def test_generated_path_used_as_fallback_when_no_core_path_exists(hub_shortcut_catalog):
    """If core/ genuinely has no path at all, the generated shortcut is
    still better than refusing to answer."""
    del hub_shortcut_catalog.relationships["b1_to_b2"]  # breaks the only core path
    del hub_shortcut_catalog._relationship_origin["b1_to_b2"]

    query = build_query(hub_shortcut_catalog, dataset="fact_a", metrics=["fact_b_count"])

    assert query.datasets_used == ["fact_a", "fact_b", "hub"]


# --- Real catalog: multi-hop join ---


def test_release_to_artist_prefers_artist_release_shortcut(real_catalog):
    """release <-> artist has two core-authored paths: the 2-hop
    artist_release materialized-table shortcut and the 3-hop
    artist_credit/artist_credit_name path. The shorter one wins — see the
    module docstring in semantic/core/release.ossie.yaml for why this
    (unusually) isn't the "prefer core over generated" tiebreak from
    test_core_path_preferred_over_shorter_generated_shortcut above, but a
    real measured performance + correctness difference between two
    hand-curated paths, found by testing against the real ~344M-row
    MusicBrainz database (the naive path took >60s and contributed to a
    Postgres shutdown under memory pressure)."""
    query = build_query(real_catalog, dataset="release", metrics=["release_count"], dimensions=["artist.name"])

    assert "FROM musicbrainz.musicbrainz.release AS release" in query.sql
    assert "LEFT JOIN musicbrainz.musicbrainz.artist_release AS artist_release ON artist_release.release = release.id" in query.sql
    assert "LEFT JOIN musicbrainz.musicbrainz.artist AS artist ON artist.id = artist_release.artist" in query.sql
    assert "artist_credit" not in query.sql
    assert "GROUP BY artist.name" in query.sql
    assert 'ORDER BY "release_count" DESC' in query.sql
    assert "LIMIT 50" in query.sql
    assert query.datasets_used == ["artist", "artist_release", "release"]


def test_release_to_artist_credit_name_path_still_reachable(real_catalog):
    """The longer artist_credit_name path isn't dead — it's still used
    when a query specifically needs artist_credit_name's own fields (e.g.
    join_phrase), which artist_release doesn't carry."""
    query = build_query(real_catalog, dataset="release", dimensions=["artist_credit_name.join_phrase"])

    assert "LEFT JOIN musicbrainz.musicbrainz.artist_credit AS artist_credit" in query.sql
    assert "LEFT JOIN musicbrainz.musicbrainz.artist_credit_name AS artist_credit_name" in query.sql


def test_single_dataset_query_has_no_joins(real_catalog):
    query = build_query(real_catalog, dataset="genre", metrics=["genre_count"])
    assert "JOIN" not in query.sql
    assert query.datasets_used == ["genre"]


def test_filter_adds_join_and_parameterizes_value(real_catalog):
    query = build_query(
        real_catalog,
        dataset="release",
        metrics=["release_count"],
        filters=[Filter(field="release_country.date_year", op=">=", value=1990)],
    )
    assert "LEFT JOIN musicbrainz.musicbrainz.release_country AS release_country" in query.sql
    assert "WHERE release_country.date_year >= %(p0)s" in query.sql
    assert query.params == {"p0": 1990}


def test_in_filter_produces_multiple_placeholders(real_catalog):
    query = build_query(
        real_catalog,
        dataset="release",
        metrics=["release_count"],
        filters=[Filter(field="release.status", op="IN", value=[1, 2, 3])],
    )
    assert "release.status IN (%(p0_0)s, %(p0_1)s, %(p0_2)s)" in query.sql
    assert query.params == {"p0_0": 1, "p0_1": 2, "p0_2": 3}


def test_dimension_only_query_uses_select_distinct_no_group_by(real_catalog):
    query = build_query(real_catalog, dataset="genre", dimensions=["genre.name"])
    assert query.sql.startswith("SELECT DISTINCT genre.name")
    assert "GROUP BY" not in query.sql
    assert 'ORDER BY "genre.name" ASC' in query.sql


def test_limit_is_clamped_to_max(real_catalog):
    query = build_query(real_catalog, dataset="genre", metrics=["genre_count"], limit=999999)
    assert "LIMIT 500" in query.sql


def test_limit_minimum_is_one(real_catalog):
    query = build_query(real_catalog, dataset="genre", metrics=["genre_count"], limit=0)
    assert "LIMIT 1" in query.sql


# --- Error paths ---


def test_unknown_dataset_raises(real_catalog):
    with pytest.raises(Exception, match="Unknown dataset"):
        build_query(real_catalog, dataset="not_a_real_dataset", metrics=["release_count"])


def test_unknown_metric_raises(real_catalog):
    with pytest.raises(Exception, match="Unknown metric"):
        build_query(real_catalog, dataset="release", metrics=["not_a_real_metric"])


def test_unqualified_dimension_raises(real_catalog):
    with pytest.raises(SqlBuildError, match="must be qualified"):
        build_query(real_catalog, dataset="release", dimensions=["name"])


def test_bad_filter_op_raises(real_catalog):
    with pytest.raises(SqlBuildError, match="Unsupported filter operator"):
        build_query(
            real_catalog,
            dataset="release",
            metrics=["release_count"],
            filters=[Filter(field="release.status", op="DROP TABLE", value=1)],
        )


def test_in_filter_requires_list_value(real_catalog):
    with pytest.raises(SqlBuildError, match="requires a list value"):
        build_query(
            real_catalog,
            dataset="release",
            metrics=["release_count"],
            filters=[Filter(field="release.status", op="IN", value=1)],
        )


def test_empty_request_raises(real_catalog):
    with pytest.raises(SqlBuildError, match="at least one metric or dimension"):
        build_query(real_catalog, dataset="release")


def test_no_join_path_raises(disconnected_catalog):
    disconnected_catalog.metrics["island_a_count"] = OSIMetric(
        name="island_a_count",
        expression=OSIExpression(
            dialects=[OSIDialectExpression(dialect=OSIDialect.ANSI_SQL, expression="COUNT(DISTINCT island_a.id)")]
        ),
    )
    with pytest.raises(SqlBuildError, match="No relationship path found"):
        build_query(
            disconnected_catalog,
            dataset="island_a",
            metrics=["island_a_count"],
            dimensions=["island_b.id"],
        )
