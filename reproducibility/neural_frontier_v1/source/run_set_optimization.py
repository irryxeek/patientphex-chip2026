"""Compare nested cross-fitted relation candidates and patient-level set decisions."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import pickle
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import patient_sets
import patientphex as pp
from run_optimization import bootstrap_difference, paper_folds, patient_score, save_pickle, write_variant


def frozen_base(prior):
    prior = Path(prior)
    config = json.loads((prior / "config.json").read_text())
    for name, digest in config["source_sha256"].items():
        pp.require(pp.digest(prior / "source" / name) == digest, f"changed prior source: {name}")
    pp.require(pp.digest(pp.__file__) == config["source_sha256"]["patientphex.py"], "reference scorer changed")
    for name, version in config["packages"].items():
        pp.require(importlib.metadata.version(name) == version, f"changed package: {name}")
    spec = importlib.util.spec_from_file_location("patientphex_frozen_enhanced", prior / "source/enhanced.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def load_pickle(path):
    # Only local checkpoints from the hash-identified prior experiment are loaded.
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def crossfit_candidates(base, ontology, documents, settings, folds, destination, forbidden=()):
    destination = Path(destination)
    indexed = pp.index_documents(documents, source=True)
    validation_ids = [d["pmc_id"] for fold in folds for d in fold]
    pp.require(len(validation_ids) == len(set(validation_ids)), "duplicate cross-fit validation document")
    pp.require(set(validation_ids) == set(indexed), "cross-fit folds must cover exactly the fitting documents")
    pp.require(not (set(indexed) & set(forbidden)), "outer validation document in cross-fit training")
    outputs, audit = [], []
    for index, validation in enumerate(folds):
        fitting = [d for i, fold in enumerate(folds) if i != index for d in fold]
        lexical = base.fit_lexical(ontology, fitting, settings)
        predictions, _ = base.predict_lexical(lexical, validation)
        pp.validate_predictions(validation, predictions, ontology)
        pp.write_json(destination / f"fold{index}/lexical.json", lexical)
        pp.write_json(destination / f"fold{index}/predictions.jsonl", predictions, True)
        audit.append(
            {
                "fold": index,
                "training": [d["pmc_id"] for d in fitting],
                "validation": [d["pmc_id"] for d in validation],
                "training_fingerprints": [pp.source_fingerprint(d) for d in fitting],
                "validation_fingerprints": [pp.source_fingerprint(d) for d in validation],
                "forbidden_outer_validation": sorted(forbidden),
            }
        )
        outputs.extend(predictions)
    pp.write_json(destination / "folds.json", audit)
    return outputs


def compact_scores(rows):
    return [
        {**{k: v for k, v in row.items() if k != "features"}, "strong_evidence": patient_sets.strong_evidence(row)}
        for row in rows
    ]


def choose(candidates):
    # Input order breaks exact ties in favor of the prior model and simpler policies.
    return max(candidates, key=lambda row: patient_score(row["metrics"]))


def score_diagnostics(documents, rows):
    gold = {
        (doc["pmc_id"], a["patient_id"]): pp.gold_association_set(a["phenotype"])
        for doc in documents
        for a in doc["association"]
    }
    labels = [int(r["concept"] in gold[r["pmc_id"], r["patient_id"]]) for r in rows]
    return {
        "candidates": len(rows),
        "positive_candidates": sum(labels),
        "positive_fraction": sum(labels) / len(labels) if labels else None,
        "brier": sum((r["probability"] - label) ** 2 for r, label in zip(rows, labels, strict=True)) / len(rows)
        if rows
        else None,
        "mean_score": sum(r["probability"] for r in rows) / len(rows) if rows else None,
        "label_policy": "literal gold compound expansion, same as primary reference evaluation",
    }


def run(args):
    started = time.monotonic()
    prior, destination = Path(args.previous_run), Path(args.output_dir)
    pp.require(not destination.exists(), f"refusing to reuse run directory: {destination}")
    base = frozen_base(prior)
    previous_config = json.loads((prior / "config.json").read_text())
    previous_selection = json.loads((prior / "selection.json").read_text())
    baseline = Path(previous_config["baseline_run"])
    inputs = json.loads((baseline / "config.json").read_text())["inputs"]
    for name, item in inputs.items():
        pp.require(pp.digest(item["path"]) == item["sha256"], f"changed original input: {name}")
    training = pp.read_jsonl(baseline / "split/train.jsonl")
    training_index = pp.index_documents(training, source=True)
    folds = [[training_index[key] for key in ids] for ids in previous_config["folds"]]
    ontology = pp.load_ontology(inputs["hpo"]["path"])
    settings = previous_selection["lexical_config"]
    pp.require(previous_selection["threshold"] == 0.2, "protocol expects the frozen threshold 0.20")
    destination.mkdir(parents=True, mode=0o700)
    (destination / "source").mkdir()
    for name in ("patientphex.py", "enhanced.py", "run_baselines.py", "analyze_baselines.py"):
        shutil.copyfile(prior / "source" / name, destination / "source" / name)
    for name in ("patient_sets.py", "run_set_optimization.py", "run_optimization.py", "predict_bundle.py"):
        shutil.copyfile(Path(__file__).with_name(name), destination / "source" / name)
    protocol = Path(__file__).with_name("SET_OPTIMIZATION_PROTOCOL.md")
    if protocol.exists():
        shutil.copyfile(protocol, destination / "protocol.md")
    policies = patient_sets.policy_grid()
    prior_models = [prior / f"inner/relation/fold{i}/relation.pkl" for i in range(len(folds))]
    prior_models.extend([prior / "outer/relation.pkl", prior / "full/relation.pkl"])
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "previous_run": str(prior.resolve()),
        "baseline_run": str(baseline.resolve()),
        "inputs": inputs,
        "fitting_sha256": pp.digest(baseline / "split/train.jsonl"),
        "development_sha256": pp.digest(baseline / "split/dev.gold.jsonl"),
        "lexical_config": settings,
        "folds": previous_config["folds"],
        "crossfit_subfolds": 3,
        "crossfit_seed": 2718,
        "full_crossfit_folds": 4,
        "full_crossfit_seed": 1729,
        "policies": policies,
        "selection": "pooled inner patient micro/macro mean; ties preserve candidate order",
        "outer_adoption": "strictly better score and patient macro no lower than prior",
        "prior_model_sha256": {str(p.resolve()): pp.digest(p) for p in prior_models},
        "source_sha256": {p.name: pp.digest(p) for p in sorted((destination / "source").iterdir())},
        "packages": previous_config["packages"],
    }
    pp.write_json(destination / "config.json", config)
    pooled_entities = []
    pooled_scores = {"in_sample": [], "crossfit": []}
    candidate_audit = []
    for index, validation in enumerate(folds):
        fitting = [d for i, fold in enumerate(folds) if i != index for d in fold]
        folder = destination / f"inner/fold{index}"
        lexical = json.loads((prior / f"inner/relation/fold{index}/lexical.json").read_text())
        valid_entities, _ = base.predict_lexical(lexical, validation)
        pooled_entities.extend(valid_entities)
        pp.write_json(folder / "validation.entities.jsonl", valid_entities, True)
        original = load_pickle(prior / f"inner/relation/fold{index}/relation.pkl")
        original_scores = base.score_relations(original, validation, valid_entities)
        saved_scores = pp.read_jsonl(prior / f"inner/relation/fold{index}/scores.jsonl")
        pp.require(
            [(r["pmc_id"], r["patient_id"], r["concept"], r["probability"]) for r in original_scores]
            == [(r["pmc_id"], r["patient_id"], r["concept"], r["probability"]) for r in saved_scores],
            "frozen inner model did not reproduce its previous scores",
        )
        pooled_scores["in_sample"].extend(original_scores)
        pp.write_json(folder / "in_sample.scores.jsonl", compact_scores(original_scores), True)
        subfolds = paper_folds(fitting, count=3, seed=2718)
        crossfit = crossfit_candidates(
            base,
            ontology,
            fitting,
            settings,
            subfolds,
            folder / "crossfit",
            forbidden={d["pmc_id"] for d in validation},
        )
        relation = base.fit_relations(fitting, crossfit, ontology)
        save_pickle(folder / "crossfit.relation.pkl", relation)
        scores = base.score_relations(relation, validation, valid_entities)
        pooled_scores["crossfit"].extend(scores)
        pp.write_json(folder / "crossfit.scores.jsonl", compact_scores(scores), True)
        original_entities, _ = base.predict_lexical(lexical, fitting, allow_training_overlap=True)
        candidate_audit.append(
            {
                "fold": index,
                "training_documents": len(fitting),
                "validation_documents": len(validation),
                "in_sample_candidate_metrics": pp.evaluate(fitting, original_entities),
                "crossfit_candidate_metrics": pp.evaluate(fitting, crossfit),
                "in_sample_relation_rows": original["training_rows"],
                "in_sample_positive_rows": original["training_positive_rows"],
                "crossfit_relation_rows": relation["training_rows"],
                "crossfit_positive_rows": relation["training_positive_rows"],
            }
        )
        print(
            json.dumps({"stage": "nested_crossfit", "fold_complete": index, "seconds": time.monotonic() - started}),
            flush=True,
        )
    pp.write_json(destination / "inner/candidate_audit.json", candidate_audit)
    pp.write_json(
        destination / "inner/score_diagnostics.json",
        {name: score_diagnostics(training, rows) for name, rows in pooled_scores.items()},
    )
    candidates = []
    for mode, scores in pooled_scores.items():
        for policy in policies:
            predictions, _ = patient_sets.apply_policy(training, pooled_entities, scores, policy)
            candidates.append({"mode": mode, "policy": policy, "metrics": pp.evaluate(training, predictions)})
    selected = choose(candidates)
    selected_sets = {
        mode: choose(
            [
                r
                for r in candidates
                if r["mode"] == mode and (r["policy"]["kind"] != "global" or r["policy"]["threshold"] == 0.2)
            ]
        )
        for mode in pooled_scores
    }
    selected_global = {
        mode: choose([r for r in candidates if r["mode"] == mode and r["policy"]["kind"] == "global"])
        for mode in pooled_scores
    }
    pp.write_json(
        destination / "selection.json",
        {
            "selected": selected,
            "set_policies": selected_sets,
            "global_policies": selected_global,
            "candidates": candidates,
            "selection_frozen_before_outer_labels_read": True,
        },
    )
    print(json.dumps({"stage": "selected", "mode": selected["mode"], "policy": selected["policy"]}), flush=True)

    # All hyperparameters and the one adoption candidate are fixed before parsing these labels.
    dev = pp.read_jsonl(baseline / "split/dev.gold.jsonl")
    outer_lexical = json.loads((prior / "outer/lexical.json").read_text())
    outer_entities, _ = base.predict_lexical(outer_lexical, dev)
    pp.require(outer_entities == pp.read_jsonl(prior / "outer/entities_only/predictions.jsonl"), "outer entity drift")
    outer_crossfit = base.fit_relations(training, pooled_entities, ontology)
    (destination / "outer").mkdir()
    save_pickle(destination / "outer/crossfit.relation.pkl", outer_crossfit)
    outer_models = {"in_sample": load_pickle(prior / "outer/relation.pkl"), "crossfit": outer_crossfit}
    outer_scores = {mode: base.score_relations(model, dev, outer_entities) for mode, model in outer_models.items()}
    baseline_policy = {"kind": "global", "threshold": 0.2}
    variants = {
        "baseline": {"mode": "in_sample", "policy": baseline_policy},
        "crossfit_only": {"mode": "crossfit", "policy": baseline_policy},
        "set_only": selected_sets["in_sample"],
        "crossfit_set": selected_sets["crossfit"],
        "in_sample_global_tuned": selected_global["in_sample"],
        "crossfit_global_tuned": selected_global["crossfit"],
        "selected": selected,
    }
    outer_metrics, diagnoses = {}, {}
    for name, item in variants.items():
        predictions, ledger = patient_sets.apply_policy(dev, outer_entities, outer_scores[item["mode"]], item["policy"])
        pp.validate_predictions(dev, predictions, ontology)
        metrics, diagnosis = write_variant(destination, f"outer/{name}", dev, predictions, ledger)
        if name == "baseline":
            pp.require(
                predictions == pp.read_jsonl(prior / "outer/complete/predictions.jsonl"), "baseline replay mismatch"
            )
        outer_metrics[name] = {
            "mode": item["mode"],
            "policy": item["policy"],
            "metrics": metrics,
            "patient_groups": patient_sets.grouped_patient_metrics(diagnosis),
        }
        diagnoses[name] = diagnosis
    paired = bootstrap_difference(diagnoses["baseline"]["documents"], diagnoses["selected"]["documents"])
    pp.write_json(destination / "outer/paired_bootstrap.json", paired)
    original_metrics, chosen_metrics = [outer_metrics[key]["metrics"] for key in ("baseline", "selected")]
    adopt = chosen_metrics["score"] > original_metrics["score"] + 1e-12 and (
        chosen_metrics["patient_macro"]["f1"] >= original_metrics["patient_macro"]["f1"] - 1e-12
    )
    decision = {
        "adopt": adopt,
        "selected_mode": selected["mode"],
        "selected_policy": selected["policy"],
        "variants": outer_metrics,
        "paired_difference": paired,
    }
    pp.write_json(destination / "outer/decision.json", decision)
    print(
        json.dumps(
            {
                "stage": "outer_result",
                "score": chosen_metrics["score"],
                "macro": chosen_metrics["patient_macro"]["f1"],
                "adopt": adopt,
            }
        ),
        flush=True,
    )
    exports = {}
    if adopt:
        full_training = pp.read_jsonl(inputs["train"]["path"])
        lexical = json.loads((prior / "full/lexical.json").read_text())
        pp.write_json(destination / "full/lexical.json", lexical)
        if selected["mode"] == "crossfit":
            full_candidates = crossfit_candidates(
                base,
                ontology,
                full_training,
                settings,
                paper_folds(full_training, count=4, seed=1729),
                destination / "full/crossfit",
            )
            relation = base.fit_relations(full_training, full_candidates, ontology)
        else:
            relation = load_pickle(prior / "full/relation.pkl")
        save_pickle(destination / "full/relation.pkl", relation)
        infer_config = {
            "relation": "patient_set_classifier",
            "threshold": selected["policy"]["threshold"],
            "lexical_file": "lexical.json",
            "relation_file": "relation.pkl",
            "set_policy": selected["policy"],
        }
        pp.write_json(destination / "full/inference_config.json", infer_config)
        pp.write_json(
            destination / "full/bundle_manifest.json",
            {
                "method": "frozen lexical matcher with cross-fit/selective patient-set classifier",
                "source_sha256": config["source_sha256"],
                "packages": config["packages"],
                "model_sha256": {
                    name: pp.digest(destination / "full" / name)
                    for name in ("lexical.json", "relation.pkl", "inference_config.json")
                },
                "fitting_data_sha256": inputs["train"]["sha256"],
                "hpo_sha256": inputs["hpo"]["sha256"],
            },
        )
        for name in ("a", "b"):
            documents = pp.read_jsonl(inputs[name]["path"])
            entities, _ = base.predict_lexical(lexical, documents)
            scores = base.score_relations(relation, documents, entities)
            predictions, evidence = patient_sets.apply_policy(documents, entities, scores, selected["policy"])
            validation = pp.validate_predictions(documents, predictions, ontology)
            output = destination / f"full/{name}.predictions.jsonl"
            pp.write_json(output, predictions, True)
            pp.write_json(destination / f"full/{name}.evidence.jsonl", evidence, True)
            exports[name] = {"path": str(output.resolve()), "sha256": pp.digest(output), "validation": validation}
    summary = {
        "state": "complete",
        "seconds": time.monotonic() - started,
        "adopt": adopt,
        "selected": {"mode": selected["mode"], "policy": selected["policy"]},
        "metrics": chosen_metrics,
        "baseline_metrics": original_metrics,
        "variants": outer_metrics,
        "paired_difference": paired,
        "selected_artifacts": str((destination / "full" if adopt else prior / "full").resolve()),
        "exports": exports,
        "online_submission": None,
    }
    pp.write_json(destination / "summary.json", summary)
    print(
        json.dumps(
            {"stage": "complete", "score": chosen_metrics["score"], "adopt": adopt, "seconds": summary["seconds"]}
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
