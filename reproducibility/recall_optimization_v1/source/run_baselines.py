"""Freeze a paper split, compare two lexical baselines and prepare A/B outputs."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import patientphex as pp


def overlap(first, second):
    return max(first[0], second[0]) < min(first[0] + first[1], second[0] + second[1])


def diagnose(gold, predictions):
    """Count disjoint error categories without using them to change predictions."""
    outputs = pp.index_documents(predictions)
    errors = Counter()
    document_rows, patient_rows, examples = [], [], []
    for doc in gold:
        pred = outputs[doc["pmc_id"]]
        expected, _ = pp.entity_units(doc, gold=True)
        actual, _ = pp.entity_units(pred, source=doc)
        matched = {unit for unit in expected & actual if unit[2] != "-1"}
        missing, extra = expected - matched, actual - matched
        for unit in sorted(missing.copy()):
            if unit[2] == "-1":
                candidates = sorted(item for item in extra if item[:2] == unit[:2])
                if candidates:
                    missing.remove(unit)
                    extra.remove(candidates[0])
        negative_spans = {(e["offset"], e["length"]) for e in pred["entities"] if pp.negative(e)}
        gold_negative_spans = {(e["offset"], e["length"]) for e in doc["entities"] if pp.negative(e)}
        for unit in sorted(missing):
            if any(unit[:2] == item[:2] for item in actual):
                category = "mention_fn_same_span_wrong_id"
            elif unit[:2] in negative_spans:
                category = "mention_fn_predicted_negative"
            elif any(overlap(unit, item) for item in actual):
                category = "mention_fn_overlapping_wrong_boundary"
            else:
                category = "mention_fn_no_overlapping_candidate"
            errors[category] += 1
            examples.append(
                {
                    "pmc_id": doc["pmc_id"],
                    "category": category,
                    "offset": unit[0],
                    "length": unit[1],
                    "identifier": unit[2],
                    "text": pp.span_text(doc, unit[0], unit[1]),
                }
            )
        for unit in sorted(extra):
            if unit[:2] in gold_negative_spans:
                category = "mention_fp_gold_negative"
            elif any(unit[:2] == item[:2] for item in expected):
                category = "mention_fp_same_span_wrong_id"
            elif any(overlap(unit, item) for item in expected):
                category = "mention_fp_overlapping_wrong_boundary"
            else:
                category = "mention_fp_no_overlapping_gold"
            errors[category] += 1
        available = {
            e["text"] if part == "-1" else part
            for e in pred["entities"]
            if not pp.negative(e) and pp.scored_entity(e, doc)
            for part in pp.identifier_parts(e["identifier"])
        }
        gold_links = {a["patient_id"]: pp.gold_association_set(a["phenotype"]) for a in doc["association"]}
        predicted_links = {a["patient_id"]: set(a["phenotype"]) for a in pred["association"]}
        for patient_id, values in gold_links.items():
            result = pp.metric(*pp.compare(values, predicted_links[patient_id]), empty_perfect=True)
            patient_rows.append({"pmc_id": doc["pmc_id"], "patient_id": patient_id, **result})
            for value in values - predicted_links[patient_id]:
                category = (
                    "patient_fn_concept_available_but_not_assigned"
                    if value in available
                    else "patient_fn_concept_not_extracted"
                )
                errors[category] += 1
            others = set().union(*(v for k, v in gold_links.items() if k != patient_id))
            for value in predicted_links[patient_id] - values:
                category = (
                    "patient_fp_other_target_patient" if value in others else "patient_fp_not_a_gold_patient_link"
                )
                errors[category] += 1
        scores = pp.evaluate([doc], [pred])
        document_rows.append(
            {
                "pmc_id": doc["pmc_id"],
                "patients": len(doc["patient"]),
                **{name: scores[name] for name in ["mention", "document", "patient_micro", "patient_macro", "score"]},
            }
        )
    return {
        "error_counts": dict(errors),
        "documents": document_rows,
        "patients": patient_rows,
        "missed_mention_examples": examples,
    }


def run(args):
    destination = Path(args.output_dir)
    pp.require(not destination.exists(), f"refusing to reuse run directory: {destination}")
    started = time.monotonic()
    documents = pp.read_jsonl(args.train)
    ontology = pp.load_ontology(args.hpo)
    audit = pp.validate_gold(documents, ontology)
    training, dev = pp.split_training(documents, args.dev_fraction, args.seed)
    # The directory and configuration exist before any validation outcome is observed.
    destination.mkdir(parents=True)
    sources = destination / "source"
    sources.mkdir()
    for filename in ["patientphex.py", "run_baselines.py"]:
        shutil.copyfile(Path(__file__).with_name(filename), sources / filename)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "selection": "highest four-component development mean; tie prefers ontology-only",
        "association_primary_policy": "split_hpo",
        "alternative_policy": "literal",
        "python": platform.python_version(),
        "source_sha256": {path.name: pp.digest(path) for path in sorted(sources.iterdir())},
        "inputs": {
            key: {"path": str(Path(value).resolve()), "sha256": pp.digest(value)}
            for key, value in {"train": args.train, "hpo": args.hpo, "a": args.a, "b": args.b}.items()
            if value
        },
        "train_pmc_ids": [d["pmc_id"] for d in training],
        "dev_pmc_ids": [d["pmc_id"] for d in dev],
        "input_labels_used_for_prediction": False,
    }
    pp.write_json(destination / "config.json", config)
    pp.write_json(destination / "gold.audit.json", audit)
    pp.write_json(destination / "split/train.jsonl", training, jsonl=True)
    pp.write_json(destination / "split/dev.gold.jsonl", dev, jsonl=True)
    variants = {}
    for name, fit in [("ontology_only", []), ("training_aliases", training)]:
        stamp = time.monotonic()
        model = pp.build_model(ontology, fit)
        predictions, evidence = pp.predict(model, dev)
        validation = pp.validate_predictions(dev, predictions, ontology)
        metrics = pp.evaluate(dev, predictions)
        literal = pp.evaluate(dev, predictions, "literal")
        diagnosis = diagnose(dev, predictions)
        output = destination / name
        for filename, value, jsonl in [
            ("model.json", model, False),
            ("dev.predictions.jsonl", predictions, True),
            ("dev.evidence.jsonl", evidence, True),
            ("metrics.json", metrics, False),
            ("metrics.literal_associations.json", literal, False),
            ("diagnosis.json", diagnosis, False),
            ("validation.json", validation, False),
        ]:
            pp.write_json(output / filename, value, jsonl)
        variants[name] = {
            "metrics": {
                key: metrics[key] for key in ["mention", "document", "patient_micro", "patient_macro", "score"]
            },
            "literal_association_score": literal["score"],
            "aliases": len(model["aliases"]),
            "training_documents": len(fit),
            "prediction_sha256": pp.digest(output / "dev.predictions.jsonl"),
            "seconds": time.monotonic() - stamp,
            "error_counts": diagnosis["error_counts"],
        }
        print(json.dumps({"variant": name, **variants[name]}, ensure_ascii=False), flush=True)
    selected = max(variants, key=lambda name: variants[name]["metrics"]["score"])
    full_model = pp.build_model(ontology, documents if selected == "training_aliases" else [])
    pp.write_json(destination / "full/model.json", full_model)
    exports = {}
    for name, path in [("a", args.a), ("b", args.b)]:
        if not path:
            continue
        inputs = pp.read_jsonl(path)
        predictions, evidence = pp.predict(full_model, inputs)
        validation = pp.validate_predictions(inputs, predictions, ontology)
        output = destination / f"full/{name}.predictions.jsonl"
        pp.write_json(output, predictions, jsonl=True)
        pp.write_json(destination / f"full/{name}.evidence.jsonl", evidence, jsonl=True)
        exports[name] = {
            "file": str(output.resolve()),
            "sha256": pp.digest(output),
            "validation": validation,
        }
    result = {
        "state": "complete",
        "evaluation_status": "local_reference_only_not_official_scorer",
        "training_documents": len(training),
        "development_documents": len(dev),
        "development_patients": sum(len(d["patient"]) for d in dev),
        "variants": variants,
        "selected": selected,
        "exports": exports,
        "total_seconds": time.monotonic() - started,
        "online_submission": None,
    }
    pp.write_json(destination / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--hpo", required=True)
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
