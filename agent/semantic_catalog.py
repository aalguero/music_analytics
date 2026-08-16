"""Merges every semantic/core/*.ossie.yaml and semantic/generated/*.ossie.yaml
file into one in-memory catalog: a flat namespace of datasets, relationships,
and metrics that agent/tools.py and agent/sql_builder.py operate on.

Datasets are deliberately spread across many files with relationships
pointing at datasets defined in *other* files (see the note at the top of
semantic/core/artist.ossie.yaml) — that's expected, not a bug, and this is
the module responsible for resolving it into one namespace. On a name
collision between core/ and generated/, core/ always wins: it's the
hand-curated, higher-quality definition; generated/ exists to fill gaps.
"""

from __future__ import annotations

import difflib
import glob
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from ossie.models import OSIDataset, OSIMetric, OSIRelationship, OSISemanticModel

DEFAULT_CORE_DIR = "semantic/core"
DEFAULT_GENERATED_DIR = "semantic/generated"


class CatalogError(Exception):
    """Raised when the merged catalog is internally inconsistent (e.g. a
    relationship or metric points at a dataset that doesn't exist anywhere)."""


@dataclass
class SemanticCatalog:
    datasets: dict[str, OSIDataset] = field(default_factory=dict)
    relationships: dict[str, OSIRelationship] = field(default_factory=dict)
    metrics: dict[str, OSIMetric] = field(default_factory=dict)
    # Which file each dataset/metric came from, for error messages and for
    # the "core beats generated" precedence rule.
    _dataset_origin: dict[str, str] = field(default_factory=dict)
    # "core" or "generated" per relationship — sql_builder.py prefers
    # hand-curated join paths over mechanically-discovered ones, since a
    # shortest-path search can't otherwise tell a genuine business
    # relationship (release -> artist_credit -> artist) apart from two
    # datasets that just happen to both reference the same lookup table for
    # unrelated reasons (e.g. release_country.country and artist.area both
    # pointing at `area`, which does NOT mean the release's country and the
    # artist's home area are the same thing).
    _relationship_origin: dict[str, str] = field(default_factory=dict)
    _metric_origin: dict[str, str] = field(default_factory=dict)

    def dataset_origin(self, name: str) -> str:
        return self._dataset_origin.get(name, "core")

    def metric_origin(self, name: str) -> str:
        return self._metric_origin.get(name, "core")

    def dataset(self, name: str) -> OSIDataset:
        try:
            return self.datasets[name]
        except KeyError:
            raise CatalogError(_unknown_name_error("dataset", name, self.datasets)) from None

    def metric(self, name: str) -> OSIMetric:
        try:
            return self.metrics[name]
        except KeyError:
            raise CatalogError(_unknown_name_error("metric", name, self.metrics)) from None

    def field(self, dataset_name: str, field_name: str):
        dataset = self.dataset(dataset_name)
        for f in dataset.fields or []:
            if f.name == field_name:
                return f
        raise CatalogError(
            f"Unknown field '{field_name}' on dataset '{dataset_name}'. "
            f"Known fields: {sorted(f.name for f in dataset.fields or [])}"
        )

    def relationships_touching(self, dataset_name: str) -> list[OSIRelationship]:
        return [r for r in self.relationships.values() if dataset_name in (r.from_dataset, r.to)]

    def relationship_origin(self, name: str) -> str:
        return self._relationship_origin.get(name, "core")

    def adjacency(self, *, origins: set[str] | None = None) -> dict[str, list[tuple[str, OSIRelationship]]]:
        """Undirected adjacency list for join-path finding: dataset name ->
        [(neighbor dataset name, relationship used to reach it), ...].

        `origins`, if given, restricts edges to relationships whose origin is
        in that set (e.g. {"core"}) — see sql_builder.py's two-phase search."""
        graph: dict[str, list[tuple[str, OSIRelationship]]] = {name: [] for name in self.datasets}
        for rel in self.relationships.values():
            if origins is not None and self.relationship_origin(rel.name) not in origins:
                continue
            graph.setdefault(rel.from_dataset, []).append((rel.to, rel))
            graph.setdefault(rel.to, []).append((rel.from_dataset, rel))
        return graph

    def summary(self) -> list[dict]:
        """One entry per dataset: name, description, and ai_context — enough
        for the agent to decide what's relevant without loading everything."""
        return [
            {
                "dataset": name,
                "description": dataset.description,
                "ai_context": ai_context_dict(dataset.ai_context),
            }
            for name, dataset in sorted(self.datasets.items())
        ]

    def metrics_summary(self) -> list[dict]:
        return [
            {"metric": name, "description": m.description, "datatype": m.datatype}
            for name, m in sorted(self.metrics.items())
        ]


def _unknown_name_error(kind: str, name: str, known: dict) -> str:
    """Error message for an unknown dataset/metric name. Suggests close
    matches instead of dumping the full name list — with 374 real datasets
    and 363 real metrics against the full MusicBrainz catalog, a bare
    "Known X: [...]" list would itself be a meaningful chunk of tokens,
    repeated on every retry."""
    suggestions = difflib.get_close_matches(name, known, n=8, cutoff=0.4)
    hint = f"Close matches: {suggestions}" if suggestions else f"({len(known)} {kind}s available; call list_semantic_models to browse)"
    return f"Unknown {kind} '{name}'. {hint}"


def ai_context_dict(ai_context) -> dict | str | None:
    if ai_context is None:
        return None
    if isinstance(ai_context, str):
        return ai_context
    return ai_context.model_dump(exclude_none=True)


def _iter_semantic_models(path: str) -> list[OSISemanticModel]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    from ossie.models import OSIDocument

    doc = OSIDocument.model_validate(raw)
    return doc.semantic_model


def load_catalog(
    core_dir: str | Path = DEFAULT_CORE_DIR,
    generated_dir: str | Path = DEFAULT_GENERATED_DIR,
) -> SemanticCatalog:
    """Load and merge semantic/core/ (authoritative) and semantic/generated/
    (fills gaps) into one SemanticCatalog. Raises CatalogError if any
    relationship or metric expression references a dataset that doesn't
    exist anywhere in the merged set.
    """
    catalog = SemanticCatalog()

    # core/ first so it always wins ties against generated/.
    for source_dir, origin in ((core_dir, "core"), (generated_dir, "generated")):
        for path in sorted(glob.glob(str(Path(source_dir) / "*.ossie.yaml"))):
            for model in _iter_semantic_models(path):
                for dataset in model.datasets:
                    if dataset.name in catalog.datasets and catalog._dataset_origin[dataset.name] == "core":
                        continue  # core/ already defined this dataset — generated/ loses
                    catalog.datasets[dataset.name] = dataset
                    catalog._dataset_origin[dataset.name] = origin
                for rel in model.relationships or []:
                    catalog.relationships[rel.name] = rel
                    catalog._relationship_origin[rel.name] = origin
                for metric in model.metrics or []:
                    catalog.metrics[metric.name] = metric
                    catalog._metric_origin[metric.name] = origin

    _validate(catalog)
    return catalog


def _validate(catalog: SemanticCatalog) -> None:
    for rel in catalog.relationships.values():
        for end in (rel.from_dataset, rel.to):
            if end not in catalog.datasets:
                raise CatalogError(f"Relationship '{rel.name}' references unknown dataset '{end}'")
