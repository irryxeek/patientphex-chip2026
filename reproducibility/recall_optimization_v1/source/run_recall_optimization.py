"""Select surface-recall changes on fitting-paper folds, then verify the fixed development set."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import lexical_recall as recall
import patient_sets
import patientphex as pp
from run_optimization import THRESHOLDS, compact_rows, save_pickle, write_bundle_manifest, write_variant
from run_set_optimization import bootstrap_difference, frozen_base, load_pickle


def run(args):
    started = time.monotonic()
    prior, destination = Path(args.previous_run), Path(args.output_dir)
    pp.require(not destination.exists(), f"refusing to reuse run directory: {destination}")
    base = frozen_base(prior)
    previous = json.loads((prior / "config.json").read_text())
    prior_selection = json.loads((prior / "selection.json").read_text())
    baseline = Path(previous["baseline_run"])
    inputs = json.loads((baseline / "config.json").read_text())["inputs"]
    for name, item in inputs.items():
        pp.require(pp.digest(item["path"]) == item["sha256"], f"changed input: {name}")
    train = pp.read_jsonl(baseline / "split/train.jsonl")
    indexed = pp.index_documents(train, source=True)
    folds = [[indexed[key] for key in ids] for ids in previous["folds"]]
    ontology = pp.load_ontology(inputs["hpo"]["path"])
    original_config = prior_selection["lexical_config"]
    pp.require(prior_selection["threshold"] == 0.2, "unexpected previous threshold")
    destination.mkdir(parents=True, mode=0o700)
    source = destination / "source"
    source.mkdir()
    for name in ("patientphex.py", "enhanced.py", "run_baselines.py", "analyze_baselines.py"):
        shutil.copyfile(prior / "source" / name, source / name)
    for name in (
        "lexical_recall.py",
        "patient_sets.py",
        "run_recall_optimization.py",
        "run_set_optimization.py",
        "run_optimization.py",
        "predict_bundle.py",
    ):
        shutil.copyfile(Path(__file__).with_name(name), source / name)
    shutil.copyfile(Path(__file__).with_name("RECALL_OPTIMIZATION_PROTOCOL.md"), destination / "protocol.md")
    grid = recall.configurations()
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "previous_run": str(prior.resolve()),
        "baseline_run": str(baseline.resolve()),
        "inputs": inputs,
        "folds": previous["folds"],
        "fitting_sha256": pp.digest(baseline / "split/train.jsonl"),
        "development_sha256": pp.digest(baseline / "split/dev.gold.jsonl"),
        "lexical_config": original_config,
        "recall_configs": grid,
        "thresholds": THRESHOLDS,
        "selection": "pooled inner four-component mean; ties prefer earlier candidate",
        "adoption": "strictly improved outer score, patient macro no lower than frozen prior",
        "source_sha256": {p.name: pp.digest(p) for p in sorted(source.iterdir())},
        "packages": previous["packages"],
    }
    pp.write_json(destination / "config.json", config)
    candidates = []
    for name, settings in grid.items():
        pooled_entities, pooled_scores = [], []
        for index, validation in enumerate(folds):
            fitting = [d for i, fold in enumerate(folds) if i != index for d in fold]
            folder = destination / f"inner/{name}/fold{index}"
            if name == "baseline":
                lexical = json.loads((prior / f"inner/relation/fold{index}/lexical.json").read_text())
                lexical["recall_config"] = {}
                relation = load_pickle(prior / f"inner/relation/fold{index}/relation.pkl")
            else:
                lexical = recall.fit_lexical(base, ontology, fitting, original_config, settings, inputs["hpo"]["path"])
                fitting_entities, _ = recall.predict_lexical(base, lexical, fitting, allow_training_overlap=True)
                relation = base.fit_relations(fitting, fitting_entities, ontology)
            entities, evidence = recall.predict_lexical(base, lexical, validation)
            scores = base.score_relations(relation, validation, entities)
            pp.validate_predictions(validation, entities, ontology)
            if name == "baseline":
                saved = pp.read_jsonl(prior / f"inner/relation/fold{index}/scores.jsonl")
                keys = ("pmc_id", "patient_id", "concept", "probability")
                pp.require(
                    [tuple(r[k] for k in keys) for r in scores] == [tuple(r[k] for k in keys) for r in saved],
                    "inner baseline did not reproduce saved scores",
                )
            pp.write_json(folder / "lexical.json", lexical)
            save_pickle(folder / "relation.pkl", relation)
            pp.write_json(folder / "entities.jsonl", entities, True)
            pp.write_json(folder / "entity_evidence.jsonl", evidence, True)
            pp.write_json(folder / "scores.jsonl", compact_rows(scores), True)
            pooled_entities.extend(entities)
            pooled_scores.extend(scores)
            print(
                json.dumps(
                    {"stage": "inner", "variant": name, "fold_complete": index, "seconds": time.monotonic() - started}
                ),
                flush=True,
            )
        for threshold in THRESHOLDS:
            predictions = base.apply_relations(train, pooled_entities, pooled_scores, threshold)
            candidates.append(
                {
                    "name": name,
                    "recall_config": settings,
                    "threshold": threshold,
                    "metrics": pp.evaluate(train, predictions),
                }
            )
        pp.write_json(destination / f"inner/{name}/metrics.json", candidates[-len(THRESHOLDS) :])
    selected = max(candidates, key=lambda c: c["metrics"]["score"])
    pp.write_json(
        destination / "selection.json",
        {"selected": selected, "candidates": candidates, "selection_frozen_before_outer_labels_read": True},
    )
    print(
        json.dumps(
            {
                "stage": "selected",
                "name": selected["name"],
                "threshold": selected["threshold"],
                "inner_score": selected["metrics"]["score"],
            }
        ),
        flush=True,
    )
    dev = pp.read_jsonl(baseline / "split/dev.gold.jsonl")
    baseline_lexical = json.loads((prior / "outer/lexical.json").read_text())
    baseline_relation = load_pickle(prior / "outer/relation.pkl")
    baseline_entities, _ = base.predict_lexical(baseline_lexical, dev)
    baseline_scores = base.score_relations(baseline_relation, dev, baseline_entities)
    baseline_predictions = base.apply_relations(dev, baseline_entities, baseline_scores, 0.2)
    pp.require(
        baseline_predictions == pp.read_jsonl(prior / "outer/complete/predictions.jsonl"), "outer baseline drift"
    )
    if selected["name"] == "baseline":
        lexical = {**baseline_lexical, "recall_config": {}}
        relation = baseline_relation
    else:
        lexical = recall.fit_lexical(
            base, ontology, train, original_config, selected["recall_config"], inputs["hpo"]["path"]
        )
        fitting_entities, _ = recall.predict_lexical(base, lexical, train, allow_training_overlap=True)
        relation = base.fit_relations(train, fitting_entities, ontology)
    pp.write_json(destination / "outer/lexical.json", lexical)
    save_pickle(destination / "outer/relation.pkl", relation)
    entities, entity_evidence = recall.predict_lexical(base, lexical, dev)
    pp.write_json(destination / "outer/entities.jsonl", entities, True)
    pp.write_json(destination / "outer/entity_evidence.jsonl", entity_evidence, True)
    scores = base.score_relations(relation, dev, entities)
    selected_predictions = base.apply_relations(dev, entities, scores, selected["threshold"])
    old_relation_scores = base.score_relations(baseline_relation, dev, entities)
    ablation = base.apply_relations(dev, entities, old_relation_scores, 0.2)
    variants = {}
    diagnoses = {}
    for name, predictions, rows in (
        ("baseline", baseline_predictions, baseline_scores),
        ("selected", selected_predictions, scores),
        ("new_entities_old_relation", ablation, old_relation_scores),
    ):
        pp.validate_predictions(dev, predictions, ontology)
        metrics, diagnosis = write_variant(destination, f"outer/{name}", dev, predictions, compact_rows(rows))
        variants[name] = {"metrics": metrics, "patient_groups": patient_sets.grouped_patient_metrics(diagnosis)}
        diagnoses[name] = diagnosis
    paired = bootstrap_difference(diagnoses["baseline"]["documents"], diagnoses["selected"]["documents"])
    old, new = [variants[name]["metrics"] for name in ("baseline", "selected")]
    adopt = new["score"] > old["score"] + 1e-12 and new["patient_macro"]["f1"] >= old["patient_macro"]["f1"] - 1e-12
    pp.write_json(
        destination / "outer/decision.json", {"adopt": adopt, "variants": variants, "paired_difference": paired}
    )
    print(
        json.dumps(
            {"stage": "outer_result", "score": new["score"], "macro": new["patient_macro"]["f1"], "adopt": adopt}
        ),
        flush=True,
    )
    exports = {}
    if adopt:
        full = pp.read_jsonl(inputs["train"]["path"])
        lexical = recall.fit_lexical(
            base, ontology, full, original_config, selected["recall_config"], inputs["hpo"]["path"]
        )
        fitting_entities, _ = recall.predict_lexical(base, lexical, full, allow_training_overlap=True)
        relation = base.fit_relations(full, fitting_entities, ontology)
        pp.write_json(destination / "full/lexical.json", lexical)
        save_pickle(destination / "full/relation.pkl", relation)
        pp.write_json(
            destination / "full/inference_config.json",
            {
                "lexical_backend": "recall",
                "relation": "classifier",
                "threshold": selected["threshold"],
                "lexical_file": "lexical.json",
                "relation_file": "relation.pkl",
            },
        )
        write_bundle_manifest(destination, config, inputs)
        for name in ("a", "b"):
            documents = pp.read_jsonl(inputs[name]["path"])
            entities, evidence = recall.predict_lexical(base, lexical, documents)
            scores = base.score_relations(relation, documents, entities)
            predictions = base.apply_relations(documents, entities, scores, selected["threshold"])
            validation = pp.validate_predictions(documents, predictions, ontology)
            output = destination / f"full/{name}.predictions.jsonl"
            pp.write_json(output, predictions, True)
            pp.write_json(destination / f"full/{name}.evidence.jsonl", compact_rows(scores), True)
            pp.write_json(destination / f"full/{name}.entity_evidence.jsonl", evidence, True)
            exports[name] = {"path": str(output.resolve()), "sha256": pp.digest(output), "validation": validation}
    summary = {
        "state": "complete",
        "seconds": time.monotonic() - started,
        "adopt": adopt,
        "selected": {key: selected[key] for key in ("name", "recall_config", "threshold")},
        "metrics": new,
        "baseline_metrics": old,
        "variants": variants,
        "paired_difference": paired,
        "selected_artifacts": str((destination / "full" if adopt else prior / "full").resolve()),
        "exports": exports,
        "online_submission": None,
    }
    pp.write_json(destination / "summary.json", summary)
    print(json.dumps({"stage": "complete", "adopt": adopt, "seconds": summary["seconds"]}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
