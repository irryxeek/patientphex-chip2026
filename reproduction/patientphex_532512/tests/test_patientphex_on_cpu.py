"""Behavioral checks for offsets, scoring, leakage and submission integrity."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "patientphex.py"
SPEC = importlib.util.spec_from_file_location("patientphex_local", SCRIPT)
pp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pp)

FEVER, SEIZURE, COUGH, OTHER = "HP:0001945", "HP:0001250", "HP:0012735", "HP:0004322"


def document(text="P1 had fever. P2 had seizures.", key="1", offset=100):
    patients = []
    for patient_id in ["P1", "P2"]:
        if patient_id in text:
            patients.append(
                {
                    "patient_id": patient_id,
                    "mention": [
                        {"text": patient_id, "offset": offset + text.index(patient_id), "length": len(patient_id)}
                    ],
                }
            )
    return {
        "pmc_id": key,
        "pmid": "pubmed-" + key,
        "patient": patients,
        "full_text": [{"section_type": "CASE REPORT", "type": "paragraph", "offset": offset, "text": text}],
        "entities": [],
        "association": [{"patient_id": p["patient_id"], "phenotype": []} for p in patients],
    }


def entity(doc, text, identifier, note=None):
    passage = next(p for p in doc["full_text"] if text in p["text"])
    return {
        "identifier": identifier,
        "type": "Phenotype",
        "offset": passage["offset"] + passage["text"].index(text),
        "length": len(text),
        "text": text,
        "note": note,
    }


def model(aliases=None):
    return {
        "schema_version": 1,
        "hpo_version": pp.HPO_VERSION,
        "hpo_sha256": "test-only",
        "training_documents": [],
        "aliases": aliases or {"fever": FEVER, "seizures": SEIZURE, "cough": COUGH},
    }


def test_hand_computed_four_component_score():
    gold = document("P1 had fever, seizures, strange gait and no cough. P2 was healthy.")
    gold["entities"] = [
        entity(gold, "fever", FEVER + ";" + OTHER),
        entity(gold, "seizures", SEIZURE),
        entity(gold, "strange gait", "-1"),
        entity(gold, "cough", COUGH, "NO"),
    ]
    gold["association"][0]["phenotype"] = [FEVER, OTHER, SEIZURE, "strange gait"]
    pred = copy.deepcopy(gold)
    pred["entities"][0]["identifier"] = FEVER
    pred["entities"][3]["note"] = None
    pred["association"][0]["phenotype"] = [FEVER, SEIZURE]
    pred["association"][1]["phenotype"] = [COUGH]
    result = pp.evaluate([gold], [pred])
    assert (result["mention"]["tp"], result["mention"]["fp"], result["mention"]["fn"]) == (3, 1, 1)
    assert result["mention"]["f1"] == pytest.approx(3 / 4)
    assert result["document"]["f1"] == pytest.approx(2 / 3)
    assert result["patient_micro"]["f1"] == pytest.approx(4 / 7)
    assert result["patient_macro"]["f1"] == pytest.approx(1 / 3)
    assert result["score"] == pytest.approx((3 / 4 + 2 / 3 + 4 / 7 + 1 / 3) / 4)


def test_gold_unmapped_mention_uses_boundaries_only():
    gold = document("P1 had strange gait.")
    gold["entities"] = [entity(gold, "strange gait", "-1")]
    pred = copy.deepcopy(gold)
    pred["entities"][0]["identifier"] = OTHER
    result = pp.evaluate([gold], [pred])
    assert result["mention"]["f1"] == 1
    assert result["document"]["fp"] == 1


def test_both_empty_patients_follow_explicit_macro_rule():
    doc = document()
    result = pp.evaluate([doc], [doc])
    assert result["patient_macro"]["f1"] == 1
    assert result["patient_micro"]["f1"] == 0


def test_marking_a_positive_entity_negative_loses_recall():
    gold = document()
    gold["entities"] = [entity(gold, "fever", FEVER)]
    pred = copy.deepcopy(gold)
    pred["entities"][0]["note"] = "NO"
    assert pp.evaluate([gold], [pred])["mention"]["fn"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "missing_doc",
        "extra_doc",
        "duplicate_doc",
        "missing_patient",
        "extra_patient",
        "duplicate_association",
        "offset",
        "text",
        "bool_offset",
        "compound_association",
    ],
)
def test_submission_integrity_failures_are_errors(change):
    source = document()
    pred = copy.deepcopy(source)
    predictions = [pred]
    if change == "missing_doc":
        predictions = []
    elif change == "extra_doc":
        predictions.append(document(key="extra"))
    elif change == "duplicate_doc":
        predictions.append(copy.deepcopy(pred))
    elif change == "missing_patient":
        pred["association"].pop()
    elif change == "extra_patient":
        pred["association"].append({"patient_id": "P3", "phenotype": []})
    elif change == "duplicate_association":
        pred["association"][0]["phenotype"] = [FEVER, FEVER]
    elif change == "compound_association":
        pred["association"][0]["phenotype"] = [FEVER + ";" + SEIZURE]
    else:
        pred["entities"] = [entity(source, "fever", FEVER)]
        if change == "offset":
            pred["entities"][0]["offset"] = 0
        elif change == "text":
            pred["entities"][0]["text"] = "FEVER"
        else:
            pred["entities"][0]["offset"] = True
    with pytest.raises(ValueError):
        pp.validate_predictions([source], predictions)


def test_unicode_offsets_are_original_character_positions():
    doc = document("İnfo: P1 had café-au-lait spots and seizures.", offset=1000)
    predictions, _ = pp.predict(model({"café au lait spots": OTHER, "seizures": SEIZURE}), [doc])
    assert [e["text"] for e in predictions[0]["entities"]] == ["café-au-lait spots", "seizures"]
    pp.validate_predictions([doc], predictions)


def test_patient_attribution_and_negation_are_local():
    doc = document("P1 had fever but no cough. P2 had seizures.")
    predictions, _ = pp.predict(model(), [doc])
    associations = {a["patient_id"]: a["phenotype"] for a in predictions[0]["association"]}
    assert associations == {"P1": [FEVER], "P2": [SEIZURE]}
    assert next(e for e in predictions[0]["entities"] if e["text"] == "cough")["note"] == "NO"


def test_generic_background_is_not_copied_to_the_only_patient():
    doc = document("P1 was examined.")
    doc["full_text"].insert(
        0,
        {
            "section_type": "INTRODUCTION",
            "type": "paragraph",
            "offset": 0,
            "text": "Seizures are usually associated with this disease.",
        },
    )
    predictions, _ = pp.predict(model(), [doc])
    assert len(predictions[0]["entities"]) == 1
    assert predictions[0]["association"][0]["phenotype"] == []


def test_shared_explicit_patients_can_receive_one_phenotype():
    doc = document("P1 and P2 had fever.")
    predictions, _ = pp.predict(model(), [doc])
    assert all(a["phenotype"] == [FEVER] for a in predictions[0]["association"])


def test_input_gold_labels_are_never_used_for_prediction():
    doc = document()
    clean, _ = pp.predict(model(), [doc])
    doc["entities"] = [{"text": "gold label must not be read"}]
    doc["association"] = [{"patient_id": "invented", "phenotype": [OTHER]}]
    poisoned, _ = pp.predict(model(), [doc])
    assert poisoned == clean


def test_matcher_does_not_cross_sentence_or_token_boundaries():
    doc = document("P1 had pseudofever. Seizures were absent.")
    predictions, _ = pp.predict(model({"fever": FEVER, "pseudofever seizures": OTHER}), [doc])
    assert predictions[0]["entities"] == []


def test_duplicate_entity_units_are_rejected():
    doc = document()
    doc["entities"] = [entity(doc, "fever", FEVER), entity(doc, "fever", FEVER + ";" + OTHER)]
    with pytest.raises(ValueError, match="duplicate or contradictory"):
        pp.validate_predictions([doc], [doc])


def test_deterministic_split_groups_same_pmid_and_identical_text():
    docs = [document(key=str(index), text=f"P1 had fever {index}.") for index in range(8)]
    docs[1]["pmid"] = docs[0]["pmid"]
    docs[2]["full_text"] = copy.deepcopy(docs[1]["full_text"])
    train, dev = pp.split_training(docs, fraction=0.3)
    assert pp.split_training(list(reversed(docs)), fraction=0.3)[1] != []
    assert {d["pmc_id"] for d in pp.split_training(list(reversed(docs)), fraction=0.3)[1]} == {d["pmc_id"] for d in dev}
    train_ids = {d["pmc_id"] for d in train}
    assert len(train_ids & {"0", "1", "2"}) in (0, 3)
    assert not train_ids & {d["pmc_id"] for d in dev}


@pytest.mark.parametrize("kind", ["pmc", "pmid", "text"])
def test_training_overlap_tripwires(kind):
    doc = document()
    fitted = model()
    fitted["training_documents"] = [
        {
            "pmc_id": doc["pmc_id"] if kind == "pmc" else "elsewhere",
            "pmid": doc["pmid"] if kind == "pmid" else "elsewhere",
            "text_sha256": pp.source_fingerprint(doc) if kind == "text" else "elsewhere",
        }
    ]
    with pytest.raises(ValueError, match="training/inference overlap"):
        pp.predict(fitted, [doc])


@pytest.fixture
def obo(tmp_path):
    path = tmp_path / "hp.obo"
    path.write_text(
        "format-version: 1.2\ndata-version: hp/releases/2026-06-23\n\n"
        "[Term]\nid: HP:0000118\nname: Phenotypic abnormality\n\n"
        f"[Term]\nid: {FEVER}\nname: Fever\nis_a: HP:0000118\n"
        'synonym: "Pyrexia" EXACT []\nsynonym: "Hot" BROAD []\n\n'
        f"[Term]\nid: {SEIZURE}\nname: Seizures\nis_a: HP:0000118\n\n"
        f"[Term]\nid: {COUGH}\nname: Cough\nis_a: HP:0000118\nis_obsolete: true\n\n"
        f"[Term]\nid: {OTHER}\nname: Outside\n",
        encoding="utf-8",
    )
    return path


def test_frozen_ontology_filters_branch_obsolete_and_broad_synonyms(obo):
    ontology = pp.load_ontology(obo)
    assert set(ontology["terms"]) == {pp.HPO_ROOT, FEVER, SEIZURE}
    fitted = pp.build_model(ontology, [])
    assert fitted["aliases"]["pyrexia"] == FEVER
    assert "hot" not in fitted["aliases"]
    obo.write_text(obo.read_text().replace("2026-06-23", "2026-09-01"))
    with pytest.raises(ValueError, match="incorrect HPO release"):
        pp.load_ontology(obo)


def test_unresolved_alias_ambiguity_is_not_invented_compound(obo):
    ontology = pp.load_ontology(obo)
    ontology["terms"][SEIZURE]["name"] = "Fever"
    fitted = pp.build_model(ontology, [])
    assert "fever" not in fitted["aliases"]


def test_cli_build_predict_validate_evaluate(obo, tmp_path, capsys):
    source, fitted, prediction = [tmp_path / name for name in ["input.jsonl", "model.json", "prediction.jsonl"]]
    doc = document()
    pp.write_json(source, [doc], jsonl=True)
    pp.main(["build", "--hpo", str(obo), "--output", str(fitted)])
    pp.main(["predict", "--model", str(fitted), "--input", str(source), "--output", str(prediction)])
    pp.main(["validate", "--hpo", str(obo), "--input", str(source), "--predictions", str(prediction)])
    assert len(pp.read_jsonl(prediction)[0]["entities"]) == 2
    metrics = tmp_path / "metrics.json"
    pp.main(["evaluate", "--gold", str(source), "--predictions", str(prediction), "--output", str(metrics)])
    assert json.loads(metrics.read_text())["evaluation_status"] == "local_reference_only_not_official_scorer"
    with pytest.raises(FileExistsError):
        pp.write_json(prediction, [])
    assert "prediction_sha256" in capsys.readouterr().out


def test_invalid_jsonl_is_not_silently_dropped(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(document()) + "\n\n")
    with pytest.raises(ValueError, match="blank JSONL line"):
        pp.read_jsonl(path)


def test_non_body_predictions_are_excluded_using_source_metadata():
    gold = document()
    gold["entities"] = [entity(gold, "fever", FEVER)]
    gold["full_text"].append({"section_type": "REFERENCES", "type": "ref", "offset": 500, "text": "Cough review."})
    pred = copy.deepcopy(gold)
    pred["entities"].append(entity(gold, "Cough", COUGH))
    # A prediction file cannot override the original passages used for scoring.
    pred["full_text"][-1]["section_type"] = "CASE REPORT"
    pred["full_text"][-1]["type"] = "paragraph"
    result = pp.evaluate([gold], [pred])
    assert result["mention"]["tp"] == 1
    assert result["mention"]["fp"] == 0
    assert result["document"]["f1"] == 1
    output, _ = pp.predict(model(), [gold])
    assert all(e["offset"] < 500 for e in output[0]["entities"])


def test_supervised_aliases_use_training_labels_only(obo):
    training = document("P1 had pyrexial spells.")
    training["entities"] = [entity(training, "pyrexial spells", FEVER)]
    training["association"][0]["phenotype"] = [FEVER]
    fitted = pp.build_model(pp.load_ontology(obo), [training])
    inference = document("P1 had pyrexial spells again.", key="2")
    output, _ = pp.predict(fitted, [inference])
    assert output[0]["entities"][0]["identifier"] == FEVER
    assert output[0]["association"][0]["phenotype"] == [FEVER]


def test_full_holdout_command_saves_reproducible_run(obo, tmp_path, capsys):
    docs = [document(f"P1 had fever on visit {index}.", key=str(index)) for index in range(5)]
    for doc in docs:
        doc["entities"] = [entity(doc, "fever", FEVER)]
        doc["association"][0]["phenotype"] = [FEVER]
    training = tmp_path / "train.jsonl"
    destination = tmp_path / "run"
    pp.write_json(training, docs, jsonl=True)
    arguments = ["holdout", "--train", str(training), "--hpo", str(obo), "--output-dir", str(destination)]
    pp.main(arguments)
    metrics = json.loads((destination / "metrics.json").read_text())
    assert not set(metrics["train_pmc_ids"]) & set(metrics["dev_pmc_ids"])
    assert len(metrics["train_pmc_ids"]) + len(metrics["dev_pmc_ids"]) == 5
    assert metrics["mention"]["fn"] == 0
    assert (destination / "dev.evidence.jsonl").exists()
    with pytest.raises(ValueError, match="refusing to reuse"):
        pp.main(arguments)
    assert "training_input_sha256" in capsys.readouterr().out


def test_heldout_split_rejects_a_single_independent_paper():
    with pytest.raises(ValueError, match="independent papers"):
        pp.split_training([document()])


def test_nullable_pmid_is_supported_and_preserved():
    doc = document()
    doc["pmid"] = None
    output, _ = pp.predict(model(), [doc])
    assert output[0]["pmid"] is None
    pp.validate_predictions([doc], output)
    del output[0]["pmid"]
    with pytest.raises(ValueError, match="missing pmid"):
        pp.validate_predictions([doc], output)


def test_gold_duplicates_and_compounds_are_audited_without_mutation():
    gold = document("P1 had fever and seizures.")
    gold["entities"] = [entity(gold, "fever", FEVER), entity(gold, "seizures", SEIZURE)]
    gold["entities"].append(copy.deepcopy(gold["entities"][0]))
    gold["association"][0]["phenotype"] = [FEVER, FEVER, FEVER + ";" + SEIZURE]
    original = copy.deepcopy(gold)
    pred, _ = pp.predict(model(), [gold])
    result = pp.evaluate([gold], pred)
    assert result["score"] == 1
    assert result["gold_audit_counts"]["duplicate_entity_unit"] == 1
    assert result["gold_audit_counts"]["duplicate_association_excess"] == 1
    literal = pp.evaluate([gold], pred, "literal")
    assert literal["patient_micro"]["tp"] == 1
    assert literal["patient_micro"]["fp"] == literal["patient_micro"]["fn"] == 1
    assert gold == original
    with pytest.raises(ValueError, match="duplicate"):
        pp.validate_predictions([gold], [gold])


def test_unresolved_gold_ids_and_auxiliary_notes_are_not_erased():
    gold = document("P1 had fever, strange gait and seizures.")
    gold["entities"] = [
        entity(gold, "fever", FEVER, "D"),
        entity(gold, "strange gait", "N"),
        entity(gold, "seizures", "MESH:D009069;HP:000328"),
    ]
    pred, _ = pp.predict(model(), [gold])
    result = pp.evaluate([gold], pred)
    assert result["mention"]["tp"] == 1
    assert result["mention"]["fn"] == 3
    assert result["document"]["fn"] == 3
    assert result["gold_audit_counts"]["nonstandard_note"] == 1
    assert result["gold_audit_counts"]["nonstandard_identifier_unit"] == 3


def test_gold_compatibility_still_rejects_broken_offsets():
    gold = document()
    gold["entities"] = [entity(gold, "fever", "N")]
    gold["entities"][0]["offset"] += 1
    with pytest.raises(ValueError, match="text disagrees"):
        pp.validate_gold([gold])


def test_mixed_mapped_unmapped_compound_uses_original_text_for_association(obo):
    gold = document("P1 had odd fever.")
    gold["entities"] = [entity(gold, "odd fever", FEVER + ";-1")]
    gold["association"][0]["phenotype"] = [FEVER, "odd fever"]
    fitted = pp.build_model(pp.load_ontology(obo), [gold])
    new = document("P1 had odd fever again.", key="new")
    predictions, _ = pp.predict(fitted, [new])
    assert set(predictions[0]["association"][0]["phenotype"]) == {FEVER, "odd fever"}
    pp.validate_predictions([new], predictions, pp.load_ontology(obo))
    replay, _ = pp.predict(fitted, [gold], allow_training_overlap=True)
    assert pp.evaluate([gold], replay)["score"] == 1


def test_ontology_declared_remapping_only_affects_training_not_gold(obo):
    obsolete, alt, uncertain = "HP:0000001", "HP:0000002", "HP:0000003"
    text = obo.read_text().replace("name: Fever\n", f"name: Fever\nalt_id: {alt}\n")
    text += (
        f"\n[Term]\nid: {obsolete}\nname: Old fever\nis_obsolete: true\nreplaced_by: {FEVER}\n"
        f"\n[Term]\nid: {uncertain}\nname: Unclear\nis_obsolete: true\nconsider: {FEVER}\n"
    )
    obo.write_text(text)
    ontology = pp.load_ontology(obo)
    assert ontology["canonical_ids"][obsolete] == ontology["canonical_ids"][alt] == FEVER
    assert uncertain not in ontology["canonical_ids"]
    gold = document("P1 had old spells and unclear spells.")
    gold["entities"] = [entity(gold, "old spells", obsolete), entity(gold, "unclear spells", uncertain)]
    original = copy.deepcopy(gold)
    fitted = pp.build_model(ontology, [gold])
    assert fitted["aliases"]["old spells"] == FEVER
    assert "unclear spells" not in fitted["aliases"]
    replay, _ = pp.predict(fitted, [gold], allow_training_overlap=True)
    metrics = pp.evaluate([gold], replay)
    assert metrics["mention"]["tp"] == 0
    assert metrics["mention"]["fn"] == 2
    assert gold == original


def test_invalid_compound_training_label_is_not_partially_learned(obo):
    gold = document("P1 had unusual spells.")
    gold["entities"] = [entity(gold, "unusual spells", FEVER + ";HP:000328")]
    fitted = pp.build_model(pp.load_ontology(obo), [gold])
    assert "unusual spells" not in fitted["aliases"]
    assert fitted["training_alias_stats"]["entities_excluded_from_alias_learning"] == 1


def test_gold_plain_text_semicolon_is_not_split_as_hpo():
    assert pp.gold_association_set(["unusual gait; recurrent falls"]) == {"unusual gait; recurrent falls"}


def test_baseline_runner_freezes_split_and_exports_complete_inputs(obo, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import analyze_baselines
    import run_baselines

    docs = [document(f"P1 had fever on visit {index}.", key=str(index)) for index in range(5)]
    for doc in docs:
        doc["entities"] = [entity(doc, "fever", FEVER)]
        doc["association"][0]["phenotype"] = [FEVER]
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    pp.write_json(train, docs, jsonl=True)
    pp.write_json(test, [document("P1 had fever on another visit.", key="test")], jsonl=True)
    result = run_baselines.run(
        SimpleNamespace(
            train=str(train),
            hpo=str(obo),
            a=str(test),
            b=None,
            output_dir=str(tmp_path / "run"),
            seed=42,
            dev_fraction=0.2,
        )
    )
    assert result["state"] == "complete"
    assert result["selected"] == "ontology_only"
    assert result["training_documents"] == 4
    assert result["development_documents"] == 1
    assert result["exports"]["a"]["validation"]["documents"] == 1
    for variant in result["variants"].values():
        assert variant["metrics"]["score"] == 1
        assert variant["error_counts"] == {}
    assert (tmp_path / "run/source/patientphex.py").exists()
    paired = analyze_baselines.analyze(tmp_path / "run", resamples=100)
    assert paired["point_estimates_reproduce_saved_metrics"]
    assert paired["comparisons"]["score"]["training_aliases_minus_ontology"]["ci95"] == [0, 0]


def test_error_diagnosis_separates_concept_recall_from_attribution(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import run_baselines

    gold = document("P1 had fever and seizures.")
    gold["entities"] = [entity(gold, "fever", FEVER), entity(gold, "seizures", SEIZURE)]
    gold["association"][0]["phenotype"] = [FEVER, SEIZURE]
    pred = copy.deepcopy(gold)
    pred["entities"].pop()
    pred["association"][0]["phenotype"] = []
    result = run_baselines.diagnose([gold], [pred])
    assert result["error_counts"] == {
        "mention_fn_no_overlapping_candidate": 1,
        "patient_fn_concept_available_but_not_assigned": 1,
        "patient_fn_concept_not_extracted": 1,
    }


def test_bootstrap_aggregation_weights_patients_not_papers(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import analyze_baselines

    rows = [
        {
            "patients": patients,
            "patient_macro": {"f1": macro},
            **{name: {"tp": 1, "fp": 0, "fn": 0} for name in ["mention", "document", "patient_micro"]},
        }
        for patients, macro in [(1, 1), (3, 0)]
    ]
    assert analyze_baselines.aggregate(rows)["patient_macro"] == 0.25
    assert analyze_baselines.aggregate(rows)["score"] == 0.8125
