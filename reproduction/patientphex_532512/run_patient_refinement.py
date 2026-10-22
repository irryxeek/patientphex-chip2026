"""Refine patient attribution with status context and patient-balanced fitting weights."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import enhanced
import patientphex as pp
from run_optimization import (
    bootstrap_difference,
    compact_rows,
    patient_score,
    save_pickle,
    write_bundle_manifest,
    write_variant,
)

THRESHOLDS = (0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)


def run(args):
    start = time.monotonic()
    prior = Path(args.previous_run)
    out = Path(args.output_dir)
    pp.require(not out.exists(), f"refusing to reuse run directory: {out}")
    previous_config = json.loads((prior / "config.json").read_text())
    previous_selection = json.loads((prior / "selection.json").read_text())
    previous_summary = json.loads((prior / "summary.json").read_text())
    baseline = Path(previous_config["baseline_run"])
    inputs = json.loads((baseline / "config.json").read_text())["inputs"]
    ontology = pp.load_ontology(inputs["hpo"]["path"])
    training = pp.read_jsonl(baseline / "split/train.jsonl")
    indexed = pp.index_documents(training, source=True)
    folds = [[indexed[key] for key in ids] for ids in previous_config["folds"]]
    name = previous_selection["lexical_name"]
    settings = previous_selection["lexical_config"]
    oof = [
        doc
        for i in range(len(folds))
        for doc in pp.read_jsonl(prior / f"inner/lexical/{name}/fold{i}.predictions.jsonl")
    ]
    out.mkdir(parents=True, mode=0o700)
    (out / "source").mkdir()
    for filename in (
        "patientphex.py",
        "enhanced.py",
        "run_patient_refinement.py",
        "run_optimization.py",
        "run_baselines.py",
        "analyze_baselines.py",
    ):
        shutil.copyfile(Path(__file__).with_name(filename), out / "source" / filename)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "previous_run": str(prior.resolve()),
        "baseline_run": str(baseline.resolve()),
        "folds": previous_config["folds"],
        "thresholds": THRESHOLDS,
        "variants": {"state_context": False, "patient_balanced": True},
        "selection": "inner patient micro/macro mean; keep previous on tie",
        "source_sha256": {p.name: pp.digest(p) for p in sorted((out / "source").iterdir())},
        "previous_summary_sha256": pp.digest(prior / "summary.json"),
        "packages": previous_config["packages"],
    }
    pp.write_json(out / "config.json", config)
    prior_threshold = previous_selection["threshold"]
    prior_metrics = (
        previous_selection["relation_inner_metrics"][str(prior_threshold)]
        if prior_threshold is not None
        else previous_selection["lexical_inner_metrics"][name]
    )
    candidates = [{"variant": "previous", "threshold": prior_threshold, "metrics": prior_metrics}]
    for variant, balanced in config["variants"].items():
        scores = []
        for index, validation in enumerate(folds):
            fitting = [d for i, fold in enumerate(folds) if i != index for d in fold]
            lexical = json.loads((prior / f"inner/relation/fold{index}/lexical.json").read_text())
            fitted, _ = enhanced.predict_lexical(lexical, fitting, allow_training_overlap=True)
            relation = enhanced.fit_relations(fitting, fitted, ontology, balance_patients=balanced)
            valid_ids = {d["pmc_id"] for d in validation}
            heldout = [d for d in oof if d["pmc_id"] in valid_ids]
            fold_scores = enhanced.score_relations(relation, validation, heldout)
            scores.extend(fold_scores)
            folder = out / f"inner/{variant}/fold{index}"
            folder.mkdir(parents=True)
            save_pickle(folder / "relation.pkl", relation)
            pp.write_json(folder / "scores.jsonl", compact_rows(fold_scores), True)
            print(
                json.dumps({"stage": variant, "fold_complete": index, "elapsed": time.monotonic() - start}), flush=True
            )
        metrics = {}
        for threshold in THRESHOLDS:
            predictions = enhanced.apply_relations(training, oof, scores, threshold)
            metrics[str(threshold)] = pp.evaluate(training, predictions)
        threshold = max(THRESHOLDS, key=lambda t: (patient_score(metrics[str(t)]), t))
        candidates.append({"variant": variant, "threshold": threshold, "metrics": metrics[str(threshold)]})
        pp.write_json(out / f"inner/{variant}/thresholds.json", metrics)
    chosen = max(candidates, key=lambda row: patient_score(row["metrics"]))
    pp.write_json(out / "selection.json", {"chosen": chosen, "candidates": candidates, "before_outer_read": True})
    print(json.dumps({"stage": "selected", "variant": chosen["variant"], "threshold": chosen["threshold"]}), flush=True)
    dev = pp.read_jsonl(baseline / "split/dev.gold.jsonl")
    entities = pp.read_jsonl(prior / "outer/entities_only/predictions.jsonl")
    if chosen["variant"] == "previous":
        predictions = pp.read_jsonl(prior / "outer/complete/predictions.jsonl")
        evidence = pp.read_jsonl(prior / "outer/complete/evidence.jsonl")
    else:
        lexical = json.loads((prior / "outer/lexical.json").read_text())
        fitted, _ = enhanced.predict_lexical(lexical, training, allow_training_overlap=True)
        relation = enhanced.fit_relations(
            training, fitted, ontology, balance_patients=config["variants"][chosen["variant"]]
        )
        (out / "outer").mkdir()
        save_pickle(out / "outer/relation.pkl", relation)
        scored = enhanced.score_relations(relation, dev, entities)
        predictions = enhanced.apply_relations(dev, entities, scored, chosen["threshold"])
        evidence = compact_rows(scored)
    pp.validate_predictions(dev, predictions, ontology)
    metrics, diagnosis = write_variant(out, "outer/complete", dev, predictions, evidence)
    old_diagnosis = json.loads((prior / "outer/complete/diagnosis.json").read_text())
    base_diagnosis = json.loads((baseline / "training_aliases/diagnosis.json").read_text())
    paired_prior = bootstrap_difference(old_diagnosis["documents"], diagnosis["documents"])
    paired_base = bootstrap_difference(base_diagnosis["documents"], diagnosis["documents"])
    pp.write_json(out / "outer/paired_previous.json", paired_prior)
    pp.write_json(out / "outer/paired_original.json", paired_base)
    adopt = metrics["score"] >= previous_summary["improved_metrics"]["score"]
    pp.write_json(out / "outer/decision.json", {"adopt": adopt, "metrics": metrics})
    print(
        json.dumps(
            {
                "stage": "outer_result",
                "score": metrics["score"],
                "patient_macro": metrics["patient_macro"]["f1"],
                "adopt": adopt,
            }
        ),
        flush=True,
    )
    if adopt and chosen["variant"] != "previous":
        documents = pp.read_jsonl(inputs["train"]["path"])
        lexical = enhanced.fit_lexical(ontology, documents, settings)
        pp.write_json(out / "full/lexical.json", lexical)
        fitted, _ = enhanced.predict_lexical(lexical, documents, allow_training_overlap=True)
        relation = enhanced.fit_relations(
            documents, fitted, ontology, balance_patients=config["variants"][chosen["variant"]]
        )
        save_pickle(out / "full/relation.pkl", relation)
        infer_config = {
            "relation": "classifier",
            "threshold": chosen["threshold"],
            "lexical_file": "lexical.json",
            "relation_file": "relation.pkl",
        }
        pp.write_json(out / "full/inference_config.json", infer_config)
        write_bundle_manifest(out, config, inputs)
        exports = {}
        for split in ("a", "b"):
            source = pp.read_jsonl(inputs[split]["path"])
            entities, _ = enhanced.predict_lexical(lexical, source)
            scored = enhanced.score_relations(relation, source, entities)
            predictions = enhanced.apply_relations(source, entities, scored, chosen["threshold"])
            validation = pp.validate_predictions(source, predictions, ontology)
            target = out / f"full/{split}.predictions.jsonl"
            pp.write_json(target, predictions, True)
            pp.write_json(out / f"full/{split}.evidence.jsonl", compact_rows(scored), True)
            exports[split] = {"path": str(target.resolve()), "sha256": pp.digest(target), "validation": validation}
        selected_artifacts = str((out / "full").resolve())
    else:
        exports = previous_summary["exports"]
        selected_artifacts = str((prior / "full").resolve())
    summary = {
        "state": "complete",
        "seconds": time.monotonic() - start,
        "chosen": chosen["variant"],
        "threshold": chosen["threshold"],
        "adopt": adopt,
        "previous_metrics": previous_summary["improved_metrics"],
        "metrics": metrics,
        "paired_previous": paired_prior,
        "paired_original": paired_base,
        "selected_artifacts": selected_artifacts,
        "exports": exports,
        "error_counts": diagnosis["error_counts"],
        "online_submission": None,
    }
    pp.write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
