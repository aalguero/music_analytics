"""Resolves a (dataset, metrics, dimensions, filters) request against a
SemanticCatalog into parameterized ANSI SQL, joined automatically by walking
the catalog's relationship graph.

Design choices, and why:

- Dataset/field/metric *names* the caller supplies are always looked up
  against the trusted catalog before being used as SQL identifiers — an
  unknown name is a SqlBuildError, never interpolated. Filter *values* are
  always passed as bind parameters, never string-formatted into the SQL
  text. Together these are what make it safe to let an LLM drive this: it
  can only ever reference things we already defined, with untrusted values
  always parameterized.
- Metric expressions (e.g. "COUNT(DISTINCT artist.id)") are stored already
  qualified with their dataset name, per how semantic/core/*.ossie.yaml and
  the generator both write them. Field expressions are stored *unqualified*
  (e.g. "name") and this module prefixes them with their dataset's alias.
  Mixing these two conventions up would silently produce wrong SQL, so
  they're handled by two different code paths below, not one.
- Every metric is an aggregate expression by construction (per the Ossie
  spec), so GROUP BY is simply "every requested dimension, whenever both
  metrics and dimensions are present." No aggregate/non-aggregate column
  analysis is needed beyond that.
- Joins are LEFT JOIN with COUNT(DISTINCT ...)-style metrics (already how
  every metric in this project is written) specifically so one-to-many joins
  (e.g. release -> release_label, which can fan out one release into several
  rows) can't silently inflate a COUNT(*)-style metric — there isn't one.

`dataset.source` strings are the three-part `database.schema.table` form
from the Ossie spec's own examples (e.g. "musicbrainz.musicbrainz.release"),
used verbatim in FROM/JOIN. This only works because Postgres accepts a
3-part reference *specifically when the leading part equals
current_database()* — confirmed directly (`cross-database references are
not implemented` otherwise) — which holds here because the real MusicBrainz
database and schema are both named "musicbrainz". If this project ever
points at a differently-named database, every `source` string needs to
change to match, or every query breaks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .semantic_catalog import SemanticCatalog

_DATASET_REF_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.")
_ALLOWED_FILTER_OPS = {"=", "!=", ">", ">=", "<", "<=", "IN", "NOT IN", "ILIKE", "NOT ILIKE"}
DEFAULT_LIMIT = 50
MAX_LIMIT = 500


class SqlBuildError(Exception):
    pass


@dataclass
class Filter:
    field: str  # "dataset.field"
    op: str
    value: object


@dataclass
class BuiltQuery:
    sql: str
    params: dict[str, object]
    datasets_used: list[str] = field(default_factory=list)


def _ansi_expr(expr) -> str:
    for d in expr.dialects:
        if d.dialect == "ANSI_SQL":
            return d.expression
    raise SqlBuildError(f"No ANSI_SQL dialect available for expression {expr!r}")


def _split_qualified(name: str, kind: str) -> tuple[str, str]:
    if "." not in name:
        raise SqlBuildError(f"{kind} '{name}' must be qualified as 'dataset.field'")
    dataset, _, field_name = name.partition(".")
    return dataset, field_name


def _referenced_datasets(expression: str, known: set[str]) -> set[str]:
    return {m for m in _DATASET_REF_RE.findall(expression) if m in known}


def _find_join_path(
    catalog: SemanticCatalog, base: str, target: str
) -> list[tuple[str, str, list[str], list[str]]]:
    """Find a join path from `base` to `target`, preferring hand-curated
    (core/) relationships over mechanically-introspected ones.

    Two-phase, not a single weighted search: first try BFS using *only*
    core-authored edges: those represent an explicit human decision that
    the connection is meaningful. Only if no such path exists, retry over
    the full graph (core + generated). This matters because plain
    shortest-path over every FK-derived edge would happily route two
    unrelated datasets through a shared lookup table (e.g. joining release
    to artist via release_country.country = area.id = artist.area — i.e.
    "the artist's home region," which has nothing to do with who made the
    release) whenever that happens to be shorter than the real path. See
    the _relationship_origin docstring in semantic_catalog.py.
    """
    if base == target:
        return []
    try:
        return _bfs(catalog, base, target, origins={"core"})
    except SqlBuildError:
        return _bfs(catalog, base, target, origins=None)


def _bfs(
    catalog: SemanticCatalog, base: str, target: str, *, origins: set[str] | None
) -> list[tuple[str, str, list[str], list[str]]]:
    """Returns the ordered list of joins needed as (new_dataset,
    joined_to_dataset, new_dataset_columns, joined_to_dataset_columns) —
    i.e. how to join `new_dataset` onto something already in the query."""
    graph = catalog.adjacency(origins=origins)
    visited = {base}
    parent: dict[str, tuple[str, object]] = {}
    queue = [base]
    while queue:
        current = queue.pop(0)
        if current == target:
            break
        for neighbor, rel in graph.get(current, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = (current, rel)
            queue.append(neighbor)
    else:
        raise SqlBuildError(f"No relationship path found from '{base}' to '{target}'")

    path = []
    node = target
    while node != base:
        prev, rel = parent[node]
        if rel.from_dataset == prev and rel.to == node:
            path.append((node, prev, rel.to_columns, rel.from_columns))
        else:
            path.append((node, prev, rel.from_columns, rel.to_columns))
        node = prev
    path.reverse()
    return path


def build_query(
    catalog: SemanticCatalog,
    *,
    dataset: str,
    metrics: list[str] | None = None,
    dimensions: list[str] | None = None,
    filters: list[Filter] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> BuiltQuery:
    metrics = metrics or []
    dimensions = dimensions or []
    filters = filters or []
    limit = max(1, min(limit, MAX_LIMIT))

    base = catalog.dataset(dataset)  # raises SqlBuildError-compatible CatalogError if unknown
    known_datasets = set(catalog.datasets)

    required: set[str] = {dataset}
    metric_exprs = []
    for metric_name in metrics:
        metric = catalog.metric(metric_name)
        expr = _ansi_expr(metric.expression)
        metric_exprs.append((metric_name, expr))
        required |= _referenced_datasets(expr, known_datasets)

    dimension_exprs = []
    for dim in dimensions:
        ds_name, field_name = _split_qualified(dim, "dimension")
        f = catalog.field(ds_name, field_name)
        dimension_exprs.append((dim, ds_name, f"{ds_name}.{_ansi_expr(f.expression)}"))
        required.add(ds_name)

    params: dict[str, object] = {}
    where_clauses = []
    for i, filt in enumerate(filters):
        if filt.op not in _ALLOWED_FILTER_OPS:
            raise SqlBuildError(f"Unsupported filter operator '{filt.op}'. Allowed: {sorted(_ALLOWED_FILTER_OPS)}")
        ds_name, field_name = _split_qualified(filt.field, "filter")
        f = catalog.field(ds_name, field_name)
        required.add(ds_name)
        column = f"{ds_name}.{_ansi_expr(f.expression)}"
        pname = f"p{i}"
        if filt.op in ("IN", "NOT IN"):
            if not isinstance(filt.value, (list, tuple)):
                raise SqlBuildError(f"Filter on '{filt.field}' with op '{filt.op}' requires a list value")
            placeholders = []
            for j, v in enumerate(filt.value):
                key = f"{pname}_{j}"
                params[key] = v
                placeholders.append(f"%({key})s")
            where_clauses.append(f"{column} {filt.op} ({', '.join(placeholders)})")
        else:
            params[pname] = filt.value
            where_clauses.append(f"{column} {filt.op} %({pname})s")

    # Build the join plan: BFS from `dataset` out to every other dataset
    # `required`, joining each new dataset onto whichever already-joined
    # dataset the path passes through.
    joined = {dataset}
    join_clauses = []
    for target in sorted(required - {dataset}):
        path = _find_join_path(catalog, dataset, target)
        for new_ds, onto_ds, new_cols, onto_cols in path:
            if new_ds in joined:
                continue
            new_dataset_obj = catalog.dataset(new_ds)
            on = " AND ".join(f"{new_ds}.{nc} = {onto_ds}.{oc}" for nc, oc in zip(new_cols, onto_cols))
            join_clauses.append(f"LEFT JOIN {new_dataset_obj.source} AS {new_ds} ON {on}")
            joined.add(new_ds)

    select_parts = [f"{expr} AS {_quote_alias(alias)}" for alias, _, expr in dimension_exprs]
    select_parts += [f"{expr} AS {_quote_alias(name)}" for name, expr in metric_exprs]
    if not select_parts:
        raise SqlBuildError("Query must request at least one metric or dimension")

    sql = [f"SELECT {', '.join(select_parts)}", f"FROM {base.source} AS {dataset}"]
    sql.extend(join_clauses)
    if where_clauses:
        sql.append("WHERE " + " AND ".join(where_clauses))
    if metrics and dimensions:
        group_by = ", ".join(expr for _, _, expr in dimension_exprs)
        sql.append(f"GROUP BY {group_by}")
    elif not metrics and dimensions:
        sql[0] = sql[0].replace("SELECT ", "SELECT DISTINCT ", 1)

    if metrics:
        sql.append(f"ORDER BY {_quote_alias(metrics[0])} DESC")
    elif dimensions:
        sql.append(f"ORDER BY {_quote_alias(dimensions[0])} ASC")
    sql.append(f"LIMIT {limit}")

    return BuiltQuery(sql="\n".join(sql), params=params, datasets_used=sorted(joined))


def _quote_alias(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
