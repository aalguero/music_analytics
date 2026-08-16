"""Auto-generates Ossie semantic model YAML for MusicBrainz tables that
aren't already hand-curated in semantic/core/. Reflects the live Postgres
schema via SQLAlchemy (columns, primary keys, foreign keys) instead of
requiring anyone to hand-maintain MusicBrainz's ~180 tables.

Foreign-key-derived relationships depend on FK constraints actually existing
in the database. MusicBrainz's own importer (admin/InitDb.pl) adds them via
CreateFKConstraints.sql for standalone/master imports, but skips it for
`--replication-type mirror` installs — switch to a mirror setup and
FK-derived relationships here silently disappear. scripts/setup_musicbrainz_db.sh
uses the (non-mirror) default, so this is safe for this project as set up.

Unlike semantic/core/, this is mechanical: no semantic judgment calls (no
synonyms, no is_time overrides, no hand-picked metrics beyond a trivial
`<table>_count`). That curation happens in semantic/core/ instead.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import click
import sqlalchemy as sa
import yaml
from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.engine import Engine

from ossie.models import (
    OSIDataset,
    OSIDataType,
    OSIDialect,
    OSIDialectExpression,
    OSIDocument,
    OSIExpression,
    OSIField,
    OSIMetric,
    OSIRelationship,
    OSISemanticModel,
)

# Column names that are reserved words in ANSI SQL and need quoting when used
# as a bare expression (MusicBrainz has a few, e.g. artist_credit_name.position).
_RESERVED_WORDS = {"position", "references", "order", "table", "column", "check"}


def _quote_ident(name: str) -> str:
    return f'"{name}"' if name.lower() in _RESERVED_WORDS else name


def _map_datatype(sa_type: sa.types.TypeEngine) -> OSIDataType | None:
    """Map a reflected SQLAlchemy column type to an Ossie DataType.

    Returns None (field omits `datatype`) for types outside the portable
    vocabulary (arrays, JSON, network types, ...), per spec guidance to omit
    rather than guess.
    """
    if isinstance(sa_type, sa.ARRAY):
        return None
    if isinstance(sa_type, sa.Boolean):
        return OSIDataType.BOOLEAN
    if isinstance(sa_type, sa.Float):  # Float subclasses Numeric — check first
        return OSIDataType.FLOAT
    if isinstance(sa_type, sa.Numeric):
        return OSIDataType.DECIMAL
    if isinstance(sa_type, sa.Integer):
        return OSIDataType.INTEGER
    if isinstance(sa_type, sa.DateTime):
        return OSIDataType.DATE_TIME_TZ if getattr(sa_type, "timezone", False) else OSIDataType.DATE_TIME
    if isinstance(sa_type, sa.Date):
        return OSIDataType.DATE
    if isinstance(sa_type, sa.Time):
        return OSIDataType.TIME
    if isinstance(sa_type, (sa.Uuid, PG_UUID, sa.String, sa.Text)):
        return OSIDataType.STRING
    return None


def core_covered_tables(core_dir: str | Path) -> set[str]:
    """Table names already hand-authored in semantic/core/*.ossie.yaml,
    identified by the last dotted segment of each dataset's `source`."""
    covered: set[str] = set()
    for path in sorted(glob.glob(str(Path(core_dir) / "*.ossie.yaml"))):
        with open(path) as f:
            raw = yaml.safe_load(f)
        for model in raw.get("semantic_model", []):
            for dataset in model.get("datasets", []):
                covered.add(dataset["source"].rsplit(".", 1)[-1])
    return covered


def _core_relationship_pairs(core_dir: str | Path) -> set[tuple[str, str]]:
    """(from, to) dataset pairs already connected by a hand-authored
    relationship somewhere in semantic/core/, stored both ways round so a
    membership check doesn't need to care which direction was declared."""
    pairs: set[tuple[str, str]] = set()
    for path in sorted(glob.glob(str(Path(core_dir) / "*.ossie.yaml"))):
        with open(path) as f:
            raw = yaml.safe_load(f)
        for model in raw.get("semantic_model", []):
            for rel in model.get("relationships", []):
                pairs.add((rel["from"], rel["to"]))
                pairs.add((rel["to"], rel["from"]))
    return pairs


def _field_for_column(col: dict) -> OSIField:
    return OSIField(
        name=col["name"],
        expression=OSIExpression(
            dialects=[OSIDialectExpression(dialect=OSIDialect.ANSI_SQL, expression=_quote_ident(col["name"]))]
        ),
        datatype=_map_datatype(col["type"]),
    )


def _relationships_for_table(inspector: sa.Inspector, table: str, schema: str) -> list[OSIRelationship]:
    relationships = []
    for fk in inspector.get_foreign_keys(table, schema=schema):
        referred_schema = fk.get("referred_schema")
        if referred_schema not in (None, schema):
            continue  # FK points outside the schema we're modeling — skip
        referred_table = fk["referred_table"]
        rel_name = f"{table}__{'_'.join(fk['constrained_columns'])}__to__{referred_table}"
        relationships.append(
            OSIRelationship(
                name=rel_name,
                from_dataset=table,
                to=referred_table,
                from_columns=list(fk["constrained_columns"]),
                to_columns=list(fk["referred_columns"]),
            )
        )
    return relationships


def _count_metric_for(table: str, pk_columns: list[str]) -> OSIMetric:
    pk_expr = ", ".join(f"{table}.{_quote_ident(c)}" for c in pk_columns)
    expr = f"COUNT(DISTINCT ({pk_expr}))" if len(pk_columns) > 1 else f"COUNT(DISTINCT {pk_expr})"
    return OSIMetric(
        name=f"{table}_count",
        expression=OSIExpression(dialects=[OSIDialectExpression(dialect=OSIDialect.ANSI_SQL, expression=expr)]),
        description=f"Total number of distinct {table} rows.",
        datatype=OSIDataType.INTEGER,
    )


@dataclass
class GeneratedTable:
    table: str
    path: Path


def generate_semantic_layer(
    engine: Engine,
    *,
    schema: str,
    source_prefix: str,
    core_dir: str | Path = "semantic/core",
    output_dir: str | Path = "semantic/generated",
) -> list[GeneratedTable]:
    """Reflect `schema` via `engine` and write one *.ossie.yaml per table not
    already covered by semantic/core/, into output_dir. Returns the tables
    that were generated (core-covered tables are skipped, not returned).

    Also writes a `_core_cross_references.ossie.yaml` (datasets: [], only
    relationships) capturing foreign keys that live *on* a core-covered
    table but point at a table this function does generate — e.g.
    artist.type -> artist_type. Those FKs are otherwise invisible: a
    core-covered table's own file/relationships are hand-authored (or
    absent), and the generated artist_type.ossie.yaml only sees FKs
    declared *on* artist_type itself, which for a pure lookup table is
    none. Without this, such relationships silently don't exist anywhere
    in the merged catalog. Discovered by running this against a schema
    with real FK constraints on core-covered tables — see semantic/core/
    docstrings for which vocabulary FKs (type/status/packaging/...) were
    deliberately left for this mechanism to pick up.

    Edges core/ already declares by hand (e.g. artist -> area) are skipped
    here even if the same FK constraint exists in the database — otherwise
    every hand-curated relationship would get a mechanically-generated
    duplicate under a different name the first time this runs against a
    schema with real FK constraints on core-covered tables.
    """
    covered = core_covered_tables(core_dir)
    already_related = _core_relationship_pairs(core_dir)
    inspector = sa.inspect(engine)
    all_tables = set(inspector.get_table_names(schema=schema))
    to_generate = sorted(all_tables - covered)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cross_ref_relationships = [
        rel
        for table in sorted(covered & all_tables)
        for rel in _relationships_for_table(inspector, table, schema)
        if (rel.from_dataset, rel.to) not in already_related
    ]
    if cross_ref_relationships:
        cross_ref_model = OSISemanticModel(
            name="musicbrainz_core_cross_references_generated",
            description=(
                "Auto-generated relationships whose `from` side is a hand-curated "
                "semantic/core/ dataset and whose `to` side is one of the tables "
                "generated alongside this file — see this function's docstring."
            ),
            datasets=[],
            relationships=cross_ref_relationships,
        )
        cross_ref_doc = OSIDocument(semantic_model=[cross_ref_model])
        (output_path / "_core_cross_references.ossie.yaml").write_text(
            "# AUTO-GENERATED by semantic/generator/introspect.py — do not hand-edit.\n"
            "# Regenerate with: python -m semantic.generator.introspect generate\n" + cross_ref_doc.to_osi_yaml()
        )

    results = []
    for table in to_generate:
        columns = inspector.get_columns(table, schema=schema)
        pk_columns = inspector.get_pk_constraint(table, schema=schema).get("constrained_columns") or []

        dataset = OSIDataset(
            name=table,
            source=f"{source_prefix}.{table}",
            primary_key=pk_columns or None,
            fields=[_field_for_column(c) for c in columns] or None,
        )
        relationships = _relationships_for_table(inspector, table, schema)
        metrics = [_count_metric_for(table, pk_columns)] if pk_columns else []

        model = OSISemanticModel(
            name=f"musicbrainz_{table}_generated",
            description=f"Auto-generated from the live schema for table `{table}` — no hand-curated core model exists for it.",
            datasets=[dataset],
            relationships=relationships or None,
            metrics=metrics or None,
        )
        doc = OSIDocument(semantic_model=[model])

        out_file = output_path / f"{table}.ossie.yaml"
        header = (
            "# AUTO-GENERATED by semantic/generator/introspect.py — do not hand-edit.\n"
            "# Regenerate with: python -m semantic.generator.introspect generate\n"
        )
        out_file.write_text(header + doc.to_osi_yaml())
        results.append(GeneratedTable(table=table, path=out_file))

    return results


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--schema", default="musicbrainz", show_default=True)
@click.option("--core-dir", default="semantic/core", show_default=True)
@click.option("--output-dir", default="semantic/generated", show_default=True)
def generate(schema: str, core_dir: str, output_dir: str) -> None:
    """Generate Ossie YAML for every table not covered by semantic/core/."""
    load_dotenv()
    url = (
        f"postgresql+psycopg://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"
    )
    engine = sa.create_engine(url)
    source_prefix = f"{os.environ['PGDATABASE']}.{schema}"
    results = generate_semantic_layer(
        engine, schema=schema, source_prefix=source_prefix, core_dir=core_dir, output_dir=output_dir
    )
    click.echo(f"Generated {len(results)} dataset file(s) into {output_dir}/")


if __name__ == "__main__":
    cli()
