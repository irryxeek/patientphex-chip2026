"""Check source offsets, conservative expansions, training isolation and recall replay."""

import copy
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_patient_sets_on_cpu import make_prior
from test_patientphex_on_cpu import document, entity
from test_patientphex_on_cpu import obo as shared_obo_fixture

ROOT = Path(__file__).resolve().parents[1]
SHORT, ANEMIA = "HP:0001156", "HP:0001903"
SETTINGS = {"min_precision": 0.5, "morphology": True, "abbreviations": True}
obo = shared_obo_fixture


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return tuple(importlib.import_module(name) for name in ("lexical_recall", "enhanced", "patientphex"))


@pytest.fixture
def recall_obo(tmp_path):
    path = tmp_path / "recall.obo"
    path.write_text(
        "format-version: 1.2\ndata-version: hp/releases/2026-06-23\n\n"
        "[Term]\nid: HP:0000118\nname: Phenotypic abnormality\n\n"
        f"[Term]\nid: {SHORT}\nname: Short fingers\nis_a: HP:0000118\n"
        'synonym: "Brachydactyly" EXACT []\nsynonym: "Brachydactylism" RELATED []\n'
        'synonym: "Shared synonym" BROAD []\n\n'
        f"[Term]\nid: {ANEMIA}\nname: Anaemia\nis_a: HP:0000118\n"
        'synonym: "Shared synonym" RELATED []\n\n'
        '[Term]\nid: HP:0000001\nname: Outside\nsynonym: "Outside synonym" RELATED []\n',
        encoding="utf-8",
    )
    return path


def fit(modules, recall_obo, settings, training=()):
    recall, base, pp = modules
    ontology = pp.load_ontology(recall_obo)
    return recall.fit_lexical(base, ontology, list(training), SETTINGS, settings, recall_obo)


def test_bag_preserves_unicode_offsets_and_ignores_gold(modules, recall_obo):
    recall, base, pp = modules
    model = fit(modules, recall_obo, {"bag": True})
    doc = document("İnfo: P1 had fingers that grew. His fingers were short.", offset=1000)
    output, evidence = recall.predict_lexical(base, model, [doc])
    assert [e["text"] for e in output[0]["entities"]] == ["fingers were short"]
    pp.validate_predictions([doc], output, pp.load_ontology(recall_obo))
    poisoned = copy.deepcopy(doc)
    poisoned["entities"], poisoned["association"] = [{"invalid": True}], [{"invalid": True}]
    assert recall.predict_lexical(base, model, [poisoned]) == (output, evidence)


@pytest.mark.parametrize(
    "text",
    [
        "P1 had fingers. Short follow-up.",
        "P1 had fingers, short follow-up.",
        "P1 had fingers\nshort follow-up.",
        "P1 had fingers not short.",
        "P1 had fingers and short follow-up.",
    ],
)
def test_bag_cannot_cross_punctuation_or_delete_polarity(modules, recall_obo, text):
    recall, base, _ = modules
    model = fit(modules, recall_obo, {"bag": True})
    predictions, _ = recall.predict_lexical(base, model, [document(text)])
    assert predictions[0]["entities"] == []


def test_surface_finds_spacing_and_spelling_variants(modules, recall_obo):
    recall, base, _ = modules
    model = fit(modules, recall_obo, {"surface": True})
    predictions, _ = recall.predict_lexical(base, model, [document("P1 had brachy-dactyly and anemia.")])
    assert {(e["text"], e["identifier"]) for e in predictions[0]["entities"]} == {
        ("brachy-dactyly", SHORT),
        ("anemia", ANEMIA),
    }


def test_extra_synonyms_require_unambiguous_in_branch_ids(modules, recall_obo):
    recall, base, _ = modules
    model = fit(modules, recall_obo, {"synonyms": True})
    assert model["recall_extra_aliases"] == {"brachydactylism": SHORT}
    predictions, _ = recall.predict_lexical(base, model, [document("P1 had brachydactylism.")])
    assert predictions[0]["entities"][0]["identifier"] == SHORT


def test_training_filter_removes_false_expansions_but_not_original(modules, recall_obo):
    recall, base, _ = modules
    fitting = document("P1 had fingers were short mentioned in a comparison.", key="fit")
    model = fit(modules, recall_obo, {"bag": True}, [fitting])
    assert model["recall_rejected_keys"]
    doc = document("P1 had fingers were short and short fingers.", key="test")
    predictions, _ = recall.predict_lexical(base, model, [doc])
    assert [e["text"] for e in predictions[0]["entities"]] == ["short fingers"]
    with pytest.raises(ValueError, match="training/inference overlap"):
        recall.predict_lexical(base, model, [fitting])


def test_longest_replacement_is_explicit(modules, recall_obo):
    recall, base, pp = modules
    training = document("P1 had fingers.", key="fit")
    training["entities"] = [entity(training, "fingers", SHORT)]
    doc = document("P1 had fingers were short.", key="test")
    # With no matching long phrase in fitting text, it remains an unseen candidate.
    model = fit(modules, recall_obo, {"bag": True}, [training])
    filled, _ = recall.predict_lexical(base, model, [doc])
    assert [e["text"] for e in filled[0]["entities"]] == ["fingers"]
    model["recall_config"]["longest"] = True
    replaced, _ = recall.predict_lexical(base, model, [doc])
    assert [e["text"] for e in replaced[0]["entities"]] == ["fingers were short"]
    pp.validate_predictions([doc], replaced, pp.load_ontology(recall_obo))


def test_recall_bundle_replays_in_a_fresh_process(modules, recall_obo, tmp_path):
    recall, base, pp = modules
    from run_optimization import compact_rows, save_pickle, write_bundle_manifest

    training = document("P1 had short fingers.", key="fit")
    training["entities"] = [entity(training, "short fingers", SHORT)]
    training["association"][0]["phenotype"] = [SHORT]
    model = fit(modules, recall_obo, {"bag": True}, [training])
    entities, _ = recall.predict_lexical(base, model, [training], allow_training_overlap=True)
    relation = base.fit_relations([training], entities, pp.load_ontology(recall_obo))
    test = [document("P1 had fingers were short.", key="test")]
    predictions, _ = recall.predict_lexical(base, model, test)
    rows = base.score_relations(relation, test, predictions)
    expected = base.apply_relations(test, predictions, rows, 0.2)
    root = tmp_path / "bundle"
    (root / "source").mkdir(parents=True)
    for name in ("patientphex.py", "enhanced.py", "lexical_recall.py"):
        shutil.copyfile(ROOT / name, root / "source" / name)
    pp.write_json(root / "full/lexical.json", model)
    save_pickle(root / "full/relation.pkl", relation)
    pp.write_json(
        root / "full/inference_config.json",
        {
            "lexical_backend": "recall",
            "relation": "classifier",
            "lexical_file": "lexical.json",
            "relation_file": "relation.pkl",
            "threshold": 0.2,
        },
    )
    pp.write_json(tmp_path / "train.jsonl", [training], True)
    pp.write_json(tmp_path / "test.jsonl", test, True)
    write_bundle_manifest(
        root,
        {
            "source_sha256": {p.name: pp.digest(p) for p in (root / "source").iterdir()},
            "packages": {
                name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
            },
        },
        {"train": {"path": str(tmp_path / "train.jsonl")}, "hpo": {"path": str(recall_obo)}},
    )
    command = [
        sys.executable,
        str(ROOT / "predict_bundle.py"),
        "--model-dir",
        str(root / "full"),
        "--input",
        str(tmp_path / "test.jsonl"),
        "--hpo",
        str(recall_obo),
        "--output",
        str(tmp_path / "replay.jsonl"),
        "--evidence",
        str(tmp_path / "evidence.jsonl"),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    assert pp.read_jsonl(tmp_path / "replay.jsonl") == expected
    assert pp.read_jsonl(tmp_path / "evidence.jsonl") == compact_rows(rows)


def test_recall_experiment_preserves_baseline_on_ties(modules, obo, tmp_path):
    _, base, pp = modules
    from run_recall_optimization import run

    prior, _ = make_prior(tmp_path, base, pp, obo)
    root = tmp_path / "recall_run"
    result = run(SimpleNamespace(previous_run=str(prior), output_dir=str(root)))
    assert result["state"] == "complete" and not result["adopt"]
    assert result["selected"]["name"] == "baseline"
    selection = json.loads((root / "selection.json").read_text())
    assert len(selection["candidates"]) == 40
    assert selection["selection_frozen_before_outer_labels_read"]
    assert result["selected_artifacts"] == str((prior / "full").resolve())
