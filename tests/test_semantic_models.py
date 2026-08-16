"""Validates every semantic/core/*.ossie.yaml file against the real Ossie
Pydantic models, and checks that every relationship's `from`/`to` dataset
name resolves to a dataset defined *somewhere* in the core/ directory (not
necessarily the same file — see the cross-file note in artist.ossie.yaml).
"""

import glob

import pytest
import yaml
from ossie.models import OSIDocument

CORE_DIR = "semantic/core"


def _load_all():
    files = sorted(glob.glob(f"{CORE_DIR}/*.ossie.yaml"))
    assert files, f"no *.ossie.yaml files found under {CORE_DIR}"
    docs = {}
    for path in files:
        with open(path) as f:
            raw = yaml.safe_load(f)
        docs[path] = OSIDocument.model_validate(raw)
    return docs


@pytest.fixture(scope="module")
def documents():
    return _load_all()


def test_each_file_parses(documents):
    for path, doc in documents.items():
        assert len(doc.semantic_model) == 1, f"{path} should have exactly one semantic_model entry"
        assert doc.semantic_model[0].datasets, f"{path} has no datasets"


def test_relationships_resolve_across_files(documents):
    dataset_names = {
        dataset.name
        for doc in documents.values()
        for dataset in doc.semantic_model[0].datasets
    }
    for path, doc in documents.items():
        for rel in doc.semantic_model[0].relationships or []:
            assert rel.from_dataset in dataset_names, f"{path}::{rel.name} references unknown dataset '{rel.from_dataset}'"
            assert rel.to in dataset_names, f"{path}::{rel.name} references unknown dataset '{rel.to}'"
