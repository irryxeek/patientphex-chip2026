"""Checks for context-bound aliases, relation leakage and grouped model selection."""

import copy
import importlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_patientphex_on_cpu import FEVER, SEIZURE, document, entity
from test_patientphex_on_cpu import obo as shared_obo_fixture

ROOT = Path(__file__).resolve().parents[1]
ELLIPTOCYTOSIS, HEART = "HP:0004445", "HP:0001635"
obo = shared_obo_fixture


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("enhanced"), importlib.import_module("patientphex")


@pytest.fixture
def ontology():
    terms = {
        FEVER: {"name": "Fever", "synonyms": ["High fever"]},
        SEIZURE: {"name": "Seizures", "synonyms": []},
        ELLIPTOCYTOSIS: {"name": "Hereditary elliptocytosis", "synonyms": []},
        HEART: {"name": "Heart failure", "synonyms": []},
    }
    return {
        "version": "2026-06-23",
        "sha256": "synthetic",
        "terms": terms,
        "canonical_ids": {key: key for key in terms},
    }


def configuration(**changes):
    return {"min_precision": 0.50, "morphology": True, "abbreviations": True, **changes}


def test_learned_short_acronym_does_not_match_pronouns(modules, ontology):
    enhanced, _ = modules
    fit = document("P1 had HE. He recovered.")
    fit["entities"] = [entity(fit, "HE", ELLIPTOCYTOSIS)]
    fitted = enhanced.fit_lexical(ontology, [fit], configuration())
    assert "he" not in fitted["aliases"]
    test = document("P1 had HE and he recovered.", key="new")
    outputs, _ = enhanced.predict_lexical(fitted, [test])
    assert outputs[0]["entities"] == []


def test_abbreviations_are_case_sensitive_and_document_scoped(modules, ontology):
    enhanced, pp = modules
    fitted = enhanced.fit_lexical(ontology, [], configuration())
    first = document("P1 had hereditary elliptocytosis (HE). He recovered but HE persisted.")
    second = document("P1 had HE in a different paper.", key="second")
    outputs, evidence = enhanced.predict_lexical(fitted, [first, second])
    assert [e["text"] for e in outputs[0]["entities"]] == ["hereditary elliptocytosis", "HE", "HE"]
    assert outputs[1]["entities"] == []
    assert evidence[0]["abbreviations"][0]["full_text"] == "hereditary elliptocytosis"
    pp.validate_predictions([first, second], outputs, ontology)


def test_conflicting_document_abbreviations_are_not_guessed(modules, ontology):
    enhanced, _ = modules
    fitted = enhanced.fit_lexical(ontology, [], configuration())
    doc = document("P1 had high fever (HF). P2 had heart failure (HF). HF was mentioned.")
    outputs, evidence = enhanced.predict_lexical(fitted, [doc])
    assert all(e["text"] != "HF" for e in outputs[0]["entities"])
    assert evidence[0]["abbreviations"] == []


def test_plural_matching_preserves_unicode_offsets(modules, ontology):
    enhanced, pp = modules
    fitted = enhanced.fit_lexical(ontology, [], configuration())
    doc = document("İnfo: P1 had a seizure.", offset=1000)
    outputs, _ = enhanced.predict_lexical(fitted, [doc])
    value = outputs[0]["entities"][0]
    assert value["text"] == "seizure"
    assert value["identifier"] == SEIZURE
    assert value["offset"] == 1000 + doc["full_text"][0]["text"].index("seizure")
    pp.validate_predictions([doc], outputs, ontology)


def test_training_context_precision_filters_unreliable_alias(modules, ontology):
    enhanced, _ = modules
    fit = document("P1 had a mass. Body mass was recorded. Mass is measured. Mass remained stable.")
    fit["entities"] = [entity(fit, "mass", HEART)]
    fitted = enhanced.fit_lexical(ontology, [fit], configuration())
    assert "mass" not in fitted["aliases"]
    assert fitted["alias_filter"]["counts"]["mass"]["precision"] == 0.25


def test_extended_numbered_patient_alias_avoids_bare_numbers(modules):
    enhanced, _ = modules
    doc = document("P1: patient 1 had fever in 1 eye.")
    doc["patient"][0]["patient_id"] = "O1"
    anchors = enhanced.extended_anchors(doc, doc["full_text"][0])
    text = doc["full_text"][0]["text"]
    assert any(text[a:b] == "patient 1" for a, b, _ in anchors)
    assert all(text[a:b] != "1" for a, b, _ in anchors)


def test_abbreviation_and_relation_features_never_read_input_labels(modules, ontology):
    enhanced, _ = modules
    model = enhanced.fit_lexical(ontology, [], configuration())
    doc = document("P1 had fever. P2 had seizures.")
    before, before_evidence = enhanced.predict_lexical(model, [doc])
    features = enhanced.relation_rows([doc], before)
    poisoned = copy.deepcopy(doc)
    poisoned["entities"] = [{"identifier": "secret"}]
    poisoned["association"] = [{"patient_id": "fake", "phenotype": ["answer leak"]}]
    after, after_evidence = enhanced.predict_lexical(model, [poisoned])
    assert (after, after_evidence) == (before, before_evidence)
    assert enhanced.relation_rows([poisoned], after) == features


def test_relation_features_include_evidence_across_passages(modules, ontology):
    enhanced, _ = modules
    doc = document("P1 was examined.")
    doc["full_text"].append({"offset": 500, "text": "She had fever.", "section_type": "CASE", "type": "paragraph"})
    model = enhanced.fit_lexical(ontology, [], configuration())
    predictions, _ = enhanced.predict_lexical(model, [doc])
    rows = enhanced.relation_rows([doc], predictions)
    assert len(rows) == 1
    assert rows[0]["features"]["most_recent_patient_max"] == 1
    assert rows[0]["features"]["clinical_max"] == 1
    assert rows[0]["features"]["same_passage_max"] == 0
    assert rows[0]["evidence"][0]["offset"] == 508


def test_negated_phenotypes_are_not_relation_candidates(modules, ontology):
    enhanced, _ = modules
    doc = document("P1 had no fever but seizures.")
    model = enhanced.fit_lexical(ontology, [], configuration())
    predictions, _ = enhanced.predict_lexical(model, [doc])
    rows = enhanced.relation_rows([doc], predictions)
    assert {row["concept"] for row in rows} == {SEIZURE}


def test_patient_health_context_stops_at_other_patients(modules, ontology):
    enhanced, _ = modules
    doc = document("P1 was diagnosed with fever. P2 was healthy and unaffected.")
    model = enhanced.fit_lexical(ontology, [], configuration())
    predictions, _ = enhanced.predict_lexical(model, [doc])
    rows = {row["patient_id"]: row for row in enhanced.relation_rows([doc], predictions)}
    assert rows["P1"]["features"]["patient_healthy_any"] == 0
    assert rows["P1"]["features"]["patient_affected_any"] == 1
    assert rows["P2"]["features"]["patient_healthy_any"] == 1
    assert rows["P2"]["features"]["patient_affected_any"] == 0


def test_patient_weights_offset_candidate_count_imbalance(modules, ontology):
    enhanced, _ = modules
    docs = [document("P1 had fever and seizures.", key="a"), document("P1 had fever.", key="b")]
    for doc in docs:
        doc["association"][0]["phenotype"] = [FEVER]
    lexical = enhanced.fit_lexical(ontology, [], configuration())
    predictions, _ = enhanced.predict_lexical(lexical, docs)
    relation = enhanced.fit_relations(docs, predictions, ontology, balance_patients=True)
    assert relation["balance_patients"]
    assert relation["training_weight_range"] == [0.75, 1.5]


def test_relation_fit_reload_and_training_overlap_checks(modules, ontology):
    enhanced, _ = modules
    fit = document("P1 had fever. P2 had seizures.")
    fit["entities"] = [entity(fit, "fever", FEVER), entity(fit, "seizures", SEIZURE)]
    fit["association"][0]["phenotype"] = [FEVER]
    fit["association"][1]["phenotype"] = [SEIZURE]
    lexical = enhanced.fit_lexical(ontology, [], configuration())
    predictions, _ = enhanced.predict_lexical(lexical, [fit])
    relation = enhanced.fit_relations([fit], predictions, ontology)
    assert relation["training_rows"] == 4
    assert relation["training_positive_rows"] == 2
    assert not any("HP:" in name or "P1" in name for name in relation["feature_names"])
    with pytest.raises(ValueError, match="training/inference overlap"):
        enhanced.score_relations(relation, [fit], predictions)
    test = document("P1 had fever on arrival. P2 had seizures later.", key="new")
    outputs, _ = enhanced.predict_lexical(lexical, [test])
    scores = enhanced.score_relations(relation, [test], outputs)
    loaded = pickle.loads(pickle.dumps(relation))
    assert enhanced.score_relations(loaded, [test], outputs) == scores
    result = enhanced.apply_relations([test], outputs, scores, threshold=1)
    assert all(row["phenotype"] == [] for row in result[0]["association"])
    assert result[0]["entities"] == outputs[0]["entities"]


def test_grouped_folds_keep_transitive_paper_matches_together(modules):
    from run_optimization import paper_folds

    docs = [document(f"P1 had fever on day {i}.", key=str(i)) for i in range(8)]
    docs[1]["pmid"] = docs[0]["pmid"]
    docs[2]["full_text"] = copy.deepcopy(docs[1]["full_text"])
    folds = paper_folds(docs, count=3)
    containing = [i for i, fold in enumerate(folds) if any(d["pmc_id"] in {"0", "1", "2"} for d in fold)]
    assert len(containing) == 1
    assert sum(map(len, folds)) == 8
    reversed_folds = paper_folds(list(reversed(docs)), count=3)
    assert [{d["pmc_id"] for d in fold} for fold in reversed_folds] == [{d["pmc_id"] for d in fold} for fold in folds]


def test_full_optimization_synthetic_run(modules, obo, tmp_path):
    enhanced, pp = modules
    from run_baselines import run as run_baseline
    from run_optimization import run
    from run_patient_refinement import run as refine

    docs = [document(f"P1 had fever on visit {i}.", key=str(i)) for i in range(5)]
    for doc in docs:
        doc["entities"] = [entity(doc, "fever", FEVER)]
        doc["association"][0]["phenotype"] = [FEVER]
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    pp.write_json(train, docs, True)
    pp.write_json(test, [document("P1 had fever on arrival.", key="new")], True)
    baseline = tmp_path / "baseline"
    run_baseline(
        SimpleNamespace(
            train=str(train),
            hpo=str(obo),
            a=str(test),
            b=str(test),
            output_dir=str(baseline),
            seed=42,
            dev_fraction=0.2,
        )
    )
    target = tmp_path / "improved"
    result = run(SimpleNamespace(baseline_run=str(baseline), output_dir=str(target)))
    assert result["state"] == "complete"
    assert result["improved_metrics"]["score"] == 1
    selection = json.loads((target / "selection.json").read_text())
    assert selection["selection_complete_before_outer_development_read"]
    assert result["exports"]["a"]["validation"]["patients"] == 1
    replay = tmp_path / "replay.jsonl"
    command = [
        sys.executable,
        str(ROOT / "predict_bundle.py"),
        "--model-dir",
        str(target / "full"),
        "--input",
        str(test),
        "--hpo",
        str(obo),
        "--output",
        str(replay),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=30, check=True)
    assert json.loads(completed.stdout)["prediction_sha256"] == result["exports"]["a"]["sha256"]
    assert replay.read_bytes() == (target / "full/a.predictions.jsonl").read_bytes()
    # The CLI must reject a changed frozen source before loading or creating outputs.
    frozen_source = target / "source/enhanced.py"
    original_source = frozen_source.read_bytes()
    frozen_source.write_bytes(original_source + b"\n# modified after fitting\n")
    rejected_output = tmp_path / "rejected.jsonl"
    rejected = subprocess.run(command[:-1] + [str(rejected_output)], text=True, capture_output=True, timeout=30)
    assert rejected.returncode != 0
    assert "frozen source hash mismatch" in rejected.stderr
    assert not rejected_output.exists()
    frozen_source.write_bytes(original_source)
    refined = refine(SimpleNamespace(previous_run=str(target), output_dir=str(tmp_path / "refined")))
    assert refined["metrics"]["score"] == 1
    assert refined["chosen"] == "previous"
    with pytest.raises(ValueError, match="refusing to reuse"):
        run(SimpleNamespace(baseline_run=str(baseline), output_dir=str(target)))
