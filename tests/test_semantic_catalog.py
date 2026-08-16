"""Tests semantic_catalog.py's merge/precedence/validation behavior using
small throwaway *.ossie.yaml files in tmp_path — isolated from the real
semantic/core/ content so these stay meaningful even as core/ grows.
"""

import pytest

from agent.semantic_catalog import CatalogError, load_catalog

DATASET_A = """
version: "0.2.0.dev0"
semantic_model:
  - name: model_a
    datasets:
      - name: widget
        source: testdb.testschema.widget
        primary_key: [id]
        fields:
          - name: id
            expression: {{dialects: [{{dialect: ANSI_SQL, expression: id}}]}}
            datatype: Integer
          - name: label
            expression: {{dialects: [{{dialect: ANSI_SQL, expression: "{label_expr}"}}]}}
            datatype: String
    metrics:
      - name: widget_count
        expression: {{dialects: [{{dialect: ANSI_SQL, expression: "COUNT(DISTINCT widget.id)"}}]}}
        datatype: Integer
"""

DATASET_B = """
version: "0.2.0.dev0"
semantic_model:
  - name: model_b
    datasets:
      - name: gadget
        source: testdb.testschema.gadget
        primary_key: [id]
        fields:
          - name: id
            expression: {dialects: [{dialect: ANSI_SQL, expression: id}]}
            datatype: Integer
          - name: widget_id
            expression: {dialects: [{dialect: ANSI_SQL, expression: widget_id}]}
            datatype: Integer
    relationships:
      - name: gadget_to_widget
        from: gadget
        to: widget
        from_columns: [widget_id]
        to_columns: [id]
"""

DATASET_BAD_RELATIONSHIP = """
version: "0.2.0.dev0"
semantic_model:
  - name: model_bad
    datasets:
      - name: orphan
        source: testdb.testschema.orphan
        primary_key: [id]
        fields:
          - name: id
            expression: {dialects: [{dialect: ANSI_SQL, expression: id}]}
            datatype: Integer
    relationships:
      - name: orphan_to_nowhere
        from: orphan
        to: does_not_exist
        from_columns: [id]
        to_columns: [id]
"""


def _write(dir_, name, content):
    (dir_ / name).write_text(content)


def test_merges_datasets_relationships_metrics_across_files(tmp_path):
    core = tmp_path / "core"
    core.mkdir()
    _write(core, "a.ossie.yaml", DATASET_A.format(label_expr="label"))
    _write(core, "b.ossie.yaml", DATASET_B)

    catalog = load_catalog(core_dir=core, generated_dir=tmp_path / "nonexistent_generated")

    assert set(catalog.datasets) == {"widget", "gadget"}
    assert "widget_count" in catalog.metrics
    assert "gadget_to_widget" in catalog.relationships
    # relationship declared in b.ossie.yaml resolves against a dataset from a.ossie.yaml
    rel = catalog.relationships["gadget_to_widget"]
    assert rel.to == "widget"


def test_core_wins_over_generated_on_name_collision(tmp_path):
    core = tmp_path / "core"
    generated = tmp_path / "generated"
    core.mkdir()
    generated.mkdir()
    _write(core, "a.ossie.yaml", DATASET_A.format(label_expr="curated_label_expr"))
    _write(generated, "a.ossie.yaml", DATASET_A.format(label_expr="mechanical_label_expr"))

    catalog = load_catalog(core_dir=core, generated_dir=generated)

    field = catalog.field("widget", "label")
    assert field.expression.dialects[0].expression == "curated_label_expr"


def test_generated_fills_gaps_not_covered_by_core(tmp_path):
    core = tmp_path / "core"
    generated = tmp_path / "generated"
    core.mkdir()
    generated.mkdir()
    _write(core, "a.ossie.yaml", DATASET_A.format(label_expr="label"))
    _write(generated, "b.ossie.yaml", DATASET_B)

    catalog = load_catalog(core_dir=core, generated_dir=generated)

    assert set(catalog.datasets) == {"widget", "gadget"}


def test_relationship_to_unknown_dataset_raises(tmp_path):
    core = tmp_path / "core"
    core.mkdir()
    _write(core, "bad.ossie.yaml", DATASET_BAD_RELATIONSHIP)

    with pytest.raises(CatalogError, match="does_not_exist"):
        load_catalog(core_dir=core, generated_dir=tmp_path / "nonexistent_generated")


def test_unknown_dataset_lookup_raises(tmp_path):
    core = tmp_path / "core"
    core.mkdir()
    _write(core, "a.ossie.yaml", DATASET_A.format(label_expr="label"))
    catalog = load_catalog(core_dir=core, generated_dir=tmp_path / "nonexistent_generated")

    with pytest.raises(CatalogError, match="Unknown dataset"):
        catalog.dataset("does_not_exist")


def test_unknown_dataset_error_suggests_close_matches_not_full_list(tmp_path):
    core = tmp_path / "core"
    core.mkdir()
    _write(core, "a.ossie.yaml", DATASET_A.format(label_expr="label"))
    catalog = load_catalog(core_dir=core, generated_dir=tmp_path / "nonexistent_generated")

    with pytest.raises(CatalogError) as exc_info:
        catalog.dataset("widgett")  # one letter off from "widget"

    message = str(exc_info.value)
    assert "widget" in message  # suggested as a close match
    assert "Known datasets" not in message  # not the old dump-everything format


def test_unknown_field_lookup_raises(tmp_path):
    core = tmp_path / "core"
    core.mkdir()
    _write(core, "a.ossie.yaml", DATASET_A.format(label_expr="label"))
    catalog = load_catalog(core_dir=core, generated_dir=tmp_path / "nonexistent_generated")

    with pytest.raises(CatalogError, match="Unknown field"):
        catalog.field("widget", "does_not_exist")


def test_real_project_catalog_loads_and_is_internally_consistent():
    """Sanity check against the actual semantic/core/ files in this repo —
    catches a relationship in real content pointing at a dataset that
    doesn't exist anywhere, which per-file validation (test_semantic_models.py)
    already covers, but this also exercises the merge path end to end."""
    catalog = load_catalog()
    assert "artist" in catalog.datasets
    assert "release" in catalog.datasets
    assert len(catalog.metrics) > 0
    summary = catalog.summary()
    assert any(entry["dataset"] == "artist" for entry in summary)
