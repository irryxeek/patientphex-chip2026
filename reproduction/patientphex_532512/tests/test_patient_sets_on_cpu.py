"""Regression checks for patient-set decisions, nested isolation and frozen replay."""

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
from test_patientphex_on_cpu import FEVER, SEIZURE, document, entity
from test_patientphex_on_cpu import obo as shared_obo_fixture

ROOT = Path(__file__).resolve().parents[1]
obo = shared_obo_fixture
SETTINGS = {"min_precision": 0.5, "morphology": True, "abbreviations": True}


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return tuple(importlib.import_module(name) for name in ("patient_sets", "enhanced", "patientphex"))


@pytest.fixture
def case(modules, obo):
    _, enhanced, pp = modules
    doc = document()
    lexical = enhanced.fit_lexical(pp.load_ontology(obo), [], SETTINGS)
    predictions, _ = enhanced.predict_lexical(lexical, [doc])
    rows = enhanced.relation_rows([doc], predictions)
    for row in rows:
        row["features"] = {}
        row["probability"] = 0.3
    return [doc], predictions, rows


def links(predictions):
    return {row["patient_id"]: set(row["phenotype"]) for row in predictions[0]["association"]}


def test_global_matches_frozen_decision_and_ignores_gold(modules, case):
    sets, enhanced, _ = modules
    docs, predictions, scores = case
    before = copy.deepcopy(case)
    scores[0]["probability"] = 0.19
    expected = enhanced.apply_relations(docs, predictions, scores, 0.2)
    actual, ledger = sets.apply_policy(docs, predictions, scores, {"kind": "global", "threshold": 0.2})
    assert actual == expected
    poisoned = copy.deepcopy(docs)
    poisoned[0]["entities"] = [{"invalid_gold": True}]
    poisoned[0]["association"] = [{"invalid_gold": True}]
    assert sets.apply_policy(poisoned, predictions, scores, {"kind": "global", "threshold": 0.2}) == (actual, ledger)
    assert docs == before[0] and predictions == before[1]
    assert actual[0]["entities"] == predictions[0]["entities"]


def test_evidence_keeps_shared_phenotype_and_filters_weak_links(modules, case):
    sets, _, _ = modules
    docs, predictions, rows = case
    for row in rows:
        if row["concept"] == FEVER:
            row["features"] = {"same_clause_exclusive_max": 1}
    output, ledger = sets.apply_policy(
        docs, predictions, rows, {"kind": "evidence", "threshold": 0.2, "weak_threshold": 0.6}
    )
    assert links(output) == {"P1": {FEVER}, "P2": {FEVER}}
    assert sum(row["accepted"] for row in ledger) == 2


def test_patient_gate_uses_source_support_or_high_score(modules, case):
    sets, _, _ = modules
    docs, predictions, rows = case
    for row in rows:
        if row["patient_id"] == "P2" and row["concept"] == SEIZURE:
            row["features"] = {"heading_owner_max": 1}
    policy = {"kind": "patient_gate", "threshold": 0.2, "gate": 0.6}
    output, _ = sets.apply_policy(docs, predictions, rows, policy)
    assert links(output) == {"P1": set(), "P2": {FEVER, SEIZURE}}
    next(row for row in rows if row["patient_id"] == "P1")["probability"] = 0.7
    output, _ = sets.apply_policy(docs, predictions, rows, policy)
    assert links(output) == {"P1": {FEVER, SEIZURE}, "P2": {FEVER, SEIZURE}}


def test_relative_threshold_is_per_patient(modules, case):
    sets, _, _ = modules
    docs, predictions, rows = case
    values = {("P1", FEVER): 0.9, ("P1", SEIZURE): 0.4, ("P2", FEVER): 0.4, ("P2", SEIZURE): 0.3}
    for row in rows:
        row["probability"] = values[row["patient_id"], row["concept"]]
    output, _ = sets.apply_policy(docs, predictions, rows, {"kind": "relative", "threshold": 0.2, "fraction": 0.5})
    assert links(output) == {"P1": {FEVER}, "P2": {FEVER, SEIZURE}}


def test_rules_preserved_and_empty_candidates_cover_all_patients(modules, case):
    sets, _, _ = modules
    docs, predictions, rows = case
    for row in rows:
        row["probability"] = 0
    output, _ = sets.apply_policy(docs, predictions, rows, {"kind": "rules", "threshold": 0.2})
    assert output == predictions
    output, ledger = sets.apply_policy(docs, predictions, [], {"kind": "patient_gate", "threshold": 0.2, "gate": 0.6})
    assert links(output) == {"P1": set(), "P2": set()}
    assert ledger == []


@pytest.mark.parametrize("failure", ["duplicate", "patient", "nan", "too_large", "negative"])
def test_invalid_relation_scores_rejected(modules, case, failure):
    sets, _, _ = modules
    docs, predictions, rows = case
    if failure == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif failure == "patient":
        rows[0]["patient_id"] = "missing"
    else:
        rows[0]["probability"] = {"nan": float("nan"), "too_large": 1.1, "negative": -0.1}[failure]
    with pytest.raises(ValueError):
        sets.apply_policy(docs, predictions, rows, {"kind": "global", "threshold": 0.2})


def test_crossfit_cannot_learn_heldout_aliases(modules, obo, tmp_path):
    _, enhanced, pp = modules
    from run_optimization import paper_folds
    from run_set_optimization import crossfit_candidates

    docs = [document(f"P1 had invented condition {i}.", key=str(i)) for i in range(6)]
    for i, doc in enumerate(docs):
        doc["entities"] = [entity(doc, f"invented condition {i}", FEVER)]
    ontology = pp.load_ontology(obo)
    folds = paper_folds(docs, count=3)
    predictions = crossfit_candidates(enhanced, ontology, docs, SETTINGS, folds, tmp_path / "crossfit")
    assert not any(d["entities"] for d in predictions)
    original = enhanced.fit_lexical(ontology, docs, SETTINGS)
    fitted, _ = enhanced.predict_lexical(original, docs, allow_training_overlap=True)
    assert all(d["entities"] for d in fitted)
    audit = json.loads((tmp_path / "crossfit/folds.json").read_text())
    assert all(not (set(f["training"]) & set(f["validation"])) for f in audit)
    with pytest.raises(ValueError, match="outer validation"):
        crossfit_candidates(enhanced, ontology, docs, SETTINGS, folds, tmp_path / "bad", forbidden={"0"})
    assert not (tmp_path / "bad").exists()


def make_prior(tmp_path, enhanced, pp, obo):
    """A complete small prior classifier bundle, with real fitting and saved scores."""
    from run_optimization import paper_folds, save_pickle

    docs = [document(f"P1 had fever on day {i}. P2 had seizures.", key=str(i)) for i in range(10)]
    for doc in docs:
        doc["entities"] = [entity(doc, "fever", FEVER), entity(doc, "seizures", SEIZURE)]
        doc["association"][0]["phenotype"] = [FEVER]
        doc["association"][1]["phenotype"] = [SEIZURE]
    train, dev = docs[:8], docs[8:]
    test = [document("P1 had fever on arrival. P2 had seizures yesterday.", key="new")]
    for name, rows in (("train", docs), ("test", test)):
        pp.write_json(tmp_path / f"{name}.jsonl", rows, True)
    inputs = {
        name: {"path": str(path), "sha256": pp.digest(path)}
        for name, path in (
            ("train", tmp_path / "train.jsonl"),
            ("a", tmp_path / "test.jsonl"),
            ("b", tmp_path / "test.jsonl"),
            ("hpo", obo),
        )
    }
    baseline = tmp_path / "baseline"
    pp.write_json(baseline / "config.json", {"inputs": inputs})
    pp.write_json(baseline / "split/train.jsonl", train, True)
    pp.write_json(baseline / "split/dev.gold.jsonl", dev, True)
    prior = tmp_path / "prior"
    (prior / "source").mkdir(parents=True)
    for name in ("enhanced.py", "patientphex.py", "run_baselines.py", "analyze_baselines.py"):
        shutil.copyfile(ROOT / name, prior / "source" / name)
    folds = paper_folds(train)
    config = {
        "baseline_run": str(baseline),
        "folds": [[d["pmc_id"] for d in fold] for fold in folds],
        "source_sha256": {p.name: pp.digest(p) for p in (prior / "source").iterdir()},
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
    }
    pp.write_json(prior / "config.json", config)
    pp.write_json(prior / "selection.json", {"lexical_config": SETTINGS, "threshold": 0.2})
    ontology = pp.load_ontology(obo)
    partitions = [
        (f"inner/relation/fold{i}", [d for j, f in enumerate(folds) if i != j for d in f], valid)
        for i, valid in enumerate(folds)
    ] + [("outer", train, dev), ("full", docs, test)]
    for name, fitting, valid in partitions:
        lexical = enhanced.fit_lexical(ontology, fitting, SETTINGS)
        candidates, _ = enhanced.predict_lexical(lexical, fitting, allow_training_overlap=True)
        model = enhanced.fit_relations(fitting, candidates, ontology)
        predictions, _ = enhanced.predict_lexical(lexical, valid)
        scores = enhanced.score_relations(model, valid, predictions)
        pp.write_json(prior / name / "lexical.json", lexical)
        save_pickle(prior / name / "relation.pkl", model)
        pp.write_json(prior / name / "scores.jsonl", scores, True)
        if name == "outer":
            pp.write_json(prior / name / "entities_only/predictions.jsonl", predictions, True)
            complete = enhanced.apply_relations(valid, predictions, scores, 0.2)
            pp.write_json(prior / name / "complete/predictions.jsonl", complete, True)
    return prior, inputs


def test_nested_selection_export_and_fresh_process_replay(modules, obo, tmp_path):
    _, enhanced, pp = modules
    from run_set_optimization import run

    prior, inputs = make_prior(tmp_path, enhanced, pp, obo)
    target = tmp_path / "new"
    summary = run(SimpleNamespace(previous_run=str(prior), output_dir=str(target)))
    assert summary["state"] == "complete" and summary["adopt"]
    # Constant scores cannot separate patients; explicit source support can.
    assert summary["selected"] == {
        "mode": "in_sample",
        "policy": {"kind": "evidence", "threshold": 0.2, "weak_threshold": 0.6},
    }
    assert summary["metrics"]["score"] == 1
    selection = json.loads((target / "selection.json").read_text())
    assert selection["selection_frozen_before_outer_labels_read"] and len(selection["candidates"]) == 28
    replay, evidence = tmp_path / "replay.jsonl", tmp_path / "evidence.jsonl"
    command = [
        sys.executable,
        str(ROOT / "predict_bundle.py"),
        "--model-dir",
        str(target / "full"),
        "--input",
        inputs["a"]["path"],
        "--hpo",
        str(obo),
        "--output",
        str(replay),
        "--evidence",
        str(evidence),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(completed.stdout)["prediction_sha256"] == summary["exports"]["a"]["sha256"]
    assert replay.read_bytes() == (target / "full/a.predictions.jsonl").read_bytes()
    assert evidence.read_bytes() == (target / "full/a.evidence.jsonl").read_bytes()
    frozen_source = target / "source/patient_sets.py"
    frozen_source.write_bytes(frozen_source.read_bytes() + b"\n# changed\n")
    command[command.index("--output") + 1] = str(tmp_path / "rejected.jsonl")
    failed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert failed.returncode != 0 and "frozen source hash mismatch" in failed.stderr
    assert not (tmp_path / "rejected.jsonl").exists()
