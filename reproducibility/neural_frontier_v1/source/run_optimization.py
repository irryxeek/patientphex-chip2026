"""Select improvements inside the fitting papers, then compare on frozen development papers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import analyze_baselines
import enhanced
import patientphex as pp
from run_baselines import diagnose

LEXICAL_CONFIGS = {
    "legacy": {"legacy": True},
    "filter025_abbr": {"min_precision": 0.25, "morphology": False, "abbreviations": True},
    "filter050_abbr": {"min_precision": 0.50, "morphology": False, "abbreviations": True},
    "filter025_morph_abbr": {"min_precision": 0.25, "morphology": True, "abbreviations": True},
    "filter050_morph_abbr": {"min_precision": 0.50, "morphology": True, "abbreviations": True},
}
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60)


def paper_folds(documents, count=4, seed=1729):
    groups = []
    for document in documents:
        keys = {"text:" + pp.source_fingerprint(document)}
        if document["pmid"]:
            keys.add("pmid:" + document["pmid"])
        connected = [group for group in groups if keys & group[0]]
        rows = [document]
        for group in connected:
            keys |= group[0]
            rows.extend(group[1])
            groups.remove(group)
        groups.append((keys, rows))
    pp.require(len(groups) >= count >= 2, "need at least one independent paper per fold")
    groups.sort(key=lambda g: hashlib.sha256(f"{seed}:{min(g[0])}".encode()).hexdigest())
    folds = [[] for _ in range(count)]
    for index, (_, rows) in enumerate(groups):
        folds[index % count].extend(rows)
    return folds


def save_pickle(path, value):
    with Path(path).open("xb") as handle:
        pickle.dump(value, handle, protocol=5)


def write_bundle_manifest(destination, config, inputs):
    model_dir = Path(destination) / "full"
    inference = json.loads((model_dir / "inference_config.json").read_text())
    filenames = ["inference_config.json", inference["lexical_file"]]
    if inference["relation_file"]:
        filenames.append(inference["relation_file"])
    pp.write_json(
        model_dir / "bundle_manifest.json",
        {
            "method": "filtered HPO lexical matching plus patient concept gradient boosting",
            "source_sha256": config["source_sha256"],
            "packages": config["packages"],
            "model_sha256": {name: pp.digest(model_dir / name) for name in filenames},
            "fitting_data_sha256": pp.digest(inputs["train"]["path"]),
            "hpo_sha256": pp.digest(inputs["hpo"]["path"]),
        },
    )


def compact_rows(rows):
    return [{key: value for key, value in row.items() if key != "features"} for row in rows]


def patient_score(metrics):
    return (metrics["patient_micro"]["f1"] + metrics["patient_macro"]["f1"]) / 2


def bootstrap_difference(before, after, resamples=10000, seed=42):
    before = {row["pmc_id"]: row for row in before}
    after = {row["pmc_id"]: row for row in after}
    pp.require(before.keys() == after.keys(), "bootstrap documents do not match")
    ids = sorted(before)
    base = analyze_baselines.aggregate(list(before.values()))
    new = analyze_baselines.aggregate(list(after.values()))
    samples = {key: [] for key in base}
    rng = random.Random(seed)
    for _ in range(resamples):
        chosen = rng.choices(ids, k=len(ids))
        lhs = analyze_baselines.aggregate([before[key] for key in chosen])
        rhs = analyze_baselines.aggregate([after[key] for key in chosen])
        for key in samples:
            samples[key].append(rhs[key] - lhs[key])
    return {
        "method": "paired paper percentile bootstrap",
        "resamples": resamples,
        "seed": seed,
        "documents": len(ids),
        "unit": "percentage points",
        "difference": {
            key: {
                "point": (new[key] - base[key]) * 100,
                "ci95": [analyze_baselines.quantile(values, q) * 100 for q in (0.025, 0.975)],
            }
            for key, values in samples.items()
        },
        "limits": "Development comparison; excludes scorer, annotation and model-selection uncertainty.",
    }


def write_variant(destination, name, documents, predictions, evidence):
    pp.write_json(destination / name / "predictions.jsonl", predictions, True)
    pp.write_json(destination / name / "evidence.jsonl", evidence, True)
    metrics = pp.evaluate(documents, predictions)
    pp.write_json(destination / name / "metrics.json", metrics)
    pp.write_json(
        destination / name / "metrics.literal_associations.json", pp.evaluate(documents, predictions, "literal")
    )
    diagnosis = diagnose(documents, predictions)
    pp.write_json(destination / name / "diagnosis.json", diagnosis)
    return metrics, diagnosis


def run(args):
    started = time.monotonic()
    previous = Path(args.baseline_run)
    destination = Path(args.output_dir)
    pp.require(not destination.exists(), f"refusing to reuse run directory: {destination}")
    baseline_config = json.loads((previous / "config.json").read_text())
    training = pp.read_jsonl(previous / "split/train.jsonl")
    ontology = pp.load_ontology(baseline_config["inputs"]["hpo"]["path"])
    folds = paper_folds(training)
    destination.mkdir(parents=True, mode=0o700)
    source = destination / "source"
    source.mkdir()
    for filename in (
        "patientphex.py",
        "enhanced.py",
        "run_optimization.py",
        "run_baselines.py",
        "analyze_baselines.py",
    ):
        shutil.copyfile(Path(__file__).with_name(filename), source / filename)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_run": str(previous.resolve()),
        "fitting_input_sha256": pp.digest(previous / "split/train.jsonl"),
        "outer_development_sha256": pp.digest(previous / "split/dev.gold.jsonl"),
        "folds": [[d["pmc_id"] for d in fold] for fold in folds],
        "fold_seed": 1729,
        "lexical_configs": LEXICAL_CONFIGS,
        "thresholds": THRESHOLDS,
        "lexical_selection": "pooled inner mention/document F1 mean; ties prefer earlier configuration",
        "relation_selection": "pooled inner patient micro/macro mean; threshold ties prefer larger threshold",
        "outer_gate": (
            "use inner-selected system if development score >= frozen original baseline; otherwise retain original"
        ),
        "source_sha256": {path.name: pp.digest(path) for path in sorted(source.iterdir())},
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
    }
    pp.write_json(destination / "config.json", config)
    lexical_results, pooled_predictions = {}, {}
    for name, settings in LEXICAL_CONFIGS.items():
        outputs = []
        for index, validation in enumerate(folds):
            fit = [d for i, fold in enumerate(folds) if i != index for d in fold]
            model = enhanced.fit_lexical(ontology, fit, settings)
            predictions, evidence = enhanced.predict_lexical(model, validation)
            pp.validate_predictions(validation, predictions, ontology)
            pp.write_json(destination / f"inner/lexical/{name}/fold{index}.predictions.jsonl", predictions, True)
            pp.write_json(destination / f"inner/lexical/{name}/fold{index}.evidence.jsonl", evidence, True)
            outputs.extend(predictions)
            print(
                json.dumps(
                    {
                        "stage": "inner_lexical",
                        "variant": name,
                        "fold_complete": index,
                        "elapsed": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        metrics = pp.evaluate(training, outputs)
        lexical_results[name] = metrics
        pooled_predictions[name] = outputs
        pp.write_json(destination / f"inner/lexical/{name}/metrics.json", metrics)
        print(
            json.dumps(
                {
                    "stage": "inner_lexical_result",
                    "variant": name,
                    "mention": metrics["mention"]["f1"],
                    "document": metrics["document"]["f1"],
                }
            ),
            flush=True,
        )
    lexical_name = max(
        lexical_results,
        key=lambda name: (lexical_results[name]["mention"]["f1"] + lexical_results[name]["document"]["f1"]) / 2,
    )
    lexical_config = LEXICAL_CONFIGS[lexical_name]
    selected_oof = pooled_predictions[lexical_name]
    all_relation_scores = []
    for index, validation in enumerate(folds):
        fit = [d for i, fold in enumerate(folds) if i != index for d in fold]
        lexical = enhanced.fit_lexical(ontology, fit, lexical_config)
        fitted_predictions, _ = enhanced.predict_lexical(lexical, fit, allow_training_overlap=True)
        relation = enhanced.fit_relations(fit, fitted_predictions, ontology)
        validation_ids = {d["pmc_id"] for d in validation}
        heldout = [d for d in selected_oof if d["pmc_id"] in validation_ids]
        scores = enhanced.score_relations(relation, validation, heldout)
        all_relation_scores.extend(scores)
        target = destination / f"inner/relation/fold{index}"
        target.mkdir(parents=True)
        save_pickle(target / "relation.pkl", relation)
        pp.write_json(target / "lexical.json", lexical)
        pp.write_json(target / "scores.jsonl", compact_rows(scores), True)
        print(
            json.dumps(
                {
                    "stage": "inner_relation",
                    "fold_complete": index,
                    "training_rows": relation["training_rows"],
                    "elapsed": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    relation_results = {}
    for threshold in THRESHOLDS:
        predicted = enhanced.apply_relations(training, selected_oof, all_relation_scores, threshold)
        metrics = pp.evaluate(training, predicted)
        relation_results[str(threshold)] = metrics
    best_threshold = max(THRESHOLDS, key=lambda t: (patient_score(relation_results[str(t)]), t))
    use_classifier = patient_score(relation_results[str(best_threshold)]) > patient_score(lexical_results[lexical_name])
    selection = {
        "lexical_name": lexical_name,
        "lexical_config": lexical_config,
        "relation": "classifier" if use_classifier else "original_rules",
        "threshold": best_threshold if use_classifier else None,
        "lexical_inner_metrics": lexical_results,
        "relation_inner_metrics": relation_results,
        "selection_complete_before_outer_development_read": True,
    }
    pp.write_json(destination / "selection.json", selection)
    print(
        json.dumps(
            {
                "stage": "selected",
                "lexical": lexical_name,
                "relation": selection["relation"],
                "threshold": selection["threshold"],
            }
        ),
        flush=True,
    )

    # Outer labels are first read after every configuration/threshold choice has been frozen.
    dev = pp.read_jsonl(previous / "split/dev.gold.jsonl")
    lexical = enhanced.fit_lexical(ontology, training, lexical_config)
    entity_predictions, entity_evidence = enhanced.predict_lexical(lexical, dev)
    pp.validate_predictions(dev, entity_predictions, ontology)
    pp.write_json(destination / "outer/lexical.json", lexical)
    entity_metrics, _ = write_variant(destination, "outer/entities_only", dev, entity_predictions, entity_evidence)
    if use_classifier:
        fitting_predictions, _ = enhanced.predict_lexical(lexical, training, allow_training_overlap=True)
        relation = enhanced.fit_relations(training, fitting_predictions, ontology)
        save_pickle(destination / "outer/relation.pkl", relation)
        scores = enhanced.score_relations(relation, dev, entity_predictions)
        improved = enhanced.apply_relations(dev, entity_predictions, scores, best_threshold)
        improved_evidence = compact_rows(scores)
    else:
        improved, improved_evidence = entity_predictions, entity_evidence
    pp.validate_predictions(dev, improved, ontology)
    improved_metrics, improved_diagnosis = write_variant(
        destination, "outer/complete", dev, improved, improved_evidence
    )
    old_predictions = pp.read_jsonl(previous / "training_aliases/dev.predictions.jsonl")
    baseline_metrics = pp.evaluate(dev, old_predictions)
    baseline_diagnosis = diagnose(dev, old_predictions)
    paired = bootstrap_difference(baseline_diagnosis["documents"], improved_diagnosis["documents"])
    pp.write_json(destination / "outer/paired_bootstrap.json", paired)
    adopt = improved_metrics["score"] >= baseline_metrics["score"]
    pp.write_json(
        destination / "outer/decision.json",
        {
            "adopt_improved": adopt,
            "baseline": baseline_metrics,
            "entities_only": entity_metrics,
            "complete": improved_metrics,
        },
    )
    print(
        json.dumps(
            {
                "stage": "outer_result",
                "baseline_score": baseline_metrics["score"],
                "entities_only_score": entity_metrics["score"],
                "complete_score": improved_metrics["score"],
                "adopt": adopt,
            }
        ),
        flush=True,
    )

    all_training = pp.read_jsonl(baseline_config["inputs"]["train"]["path"])
    full_lexical = enhanced.fit_lexical(ontology, all_training, lexical_config if adopt else {"legacy": True})
    pp.write_json(destination / "full/lexical.json", full_lexical)
    full_relation = None
    if adopt and use_classifier:
        full_predictions, _ = enhanced.predict_lexical(full_lexical, all_training, allow_training_overlap=True)
        full_relation = enhanced.fit_relations(all_training, full_predictions, ontology)
        save_pickle(destination / "full/relation.pkl", full_relation)
    pp.write_json(
        destination / "full/inference_config.json",
        {
            "relation": "classifier" if full_relation else "original_rules",
            "threshold": best_threshold if full_relation else None,
            "lexical_file": "lexical.json",
            "relation_file": "relation.pkl" if full_relation else None,
        },
    )
    write_bundle_manifest(destination, config, baseline_config["inputs"])
    exports = {}
    for name in ("a", "b"):
        documents = pp.read_jsonl(baseline_config["inputs"][name]["path"])
        predictions, evidence = enhanced.predict_lexical(full_lexical, documents)
        if full_relation:
            scores = enhanced.score_relations(full_relation, documents, predictions)
            predictions = enhanced.apply_relations(documents, predictions, scores, best_threshold)
            evidence = compact_rows(scores)
        validation = pp.validate_predictions(documents, predictions, ontology)
        output = destination / f"full/{name}.predictions.jsonl"
        pp.write_json(output, predictions, True)
        pp.write_json(destination / f"full/{name}.evidence.jsonl", evidence, True)
        exports[name] = {"path": str(output.resolve()), "sha256": pp.digest(output), "validation": validation}
    summary = {
        "state": "complete",
        "run_seconds": time.monotonic() - started,
        "selected_lexical": lexical_name,
        "selected_relation": selection["relation"],
        "threshold": selection["threshold"],
        "adopted_improvement": adopt,
        "baseline_metrics": baseline_metrics,
        "entities_only_metrics": entity_metrics,
        "improved_metrics": improved_metrics,
        "error_counts": improved_diagnosis["error_counts"],
        "paired_difference": paired,
        "exports": exports,
        "online_submission": None,
    }
    pp.write_json(destination / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
