"""Paired paper bootstrap for the frozen development comparison (standard library)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import patientphex as pp


def aggregate(rows):
    components = {}
    for name in ["mention", "document", "patient_micro"]:
        counts = [sum(row[name][key] for row in rows) for key in ["tp", "fp", "fn"]]
        components[name] = pp.metric(*counts)["f1"]
    patients = sum(row["patients"] for row in rows)
    components["patient_macro"] = (
        sum(row["patients"] * row["patient_macro"]["f1"] for row in rows) / patients if patients else 0.0
    )
    return {**components, "score": sum(components.values()) / 4}


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def analyze(run_dir, resamples=10000, seed=42):
    pp.require(resamples >= 100, "use at least 100 bootstrap resamples")
    root = Path(run_dir)
    names = ["ontology_only", "training_aliases"]
    indexed = {}
    for name in names:
        rows = json.loads((root / name / "diagnosis.json").read_text())["documents"]
        indexed[name] = {row["pmc_id"]: row for row in rows}
        pp.require(len(rows) == len(indexed[name]), "duplicate diagnostic document")
    ids = sorted(indexed[names[0]])
    pp.require(bool(ids) and set(ids) == set(indexed[names[1]]), "paired diagnostic documents mismatch")
    points = {name: aggregate(list(indexed[name].values())) for name in names}
    for name in names:
        saved = json.loads((root / name / "metrics.json").read_text())
        for metric, value in points[name].items():
            target = saved[metric] if metric == "score" else saved[metric]["f1"]
            pp.require(abs(value - target) < 1e-12, f"aggregate does not reproduce saved {name}/{metric}")
    samples = {name: {metric: [] for metric in points[names[0]]} for name in [*names, "difference"]}
    generator = random.Random(seed)
    for _ in range(resamples):
        selection = generator.choices(ids, k=len(ids))
        scores = {name: aggregate([indexed[name][key] for key in selection]) for name in names}
        for metric in points[names[0]]:
            for name in names:
                samples[name][metric].append(scores[name][metric])
            samples["difference"][metric].append(scores[names[1]][metric] - scores[names[0]][metric])
    result = {
        "method": "paired document-cluster percentile bootstrap",
        "resamples": resamples,
        "seed": seed,
        "documents": len(ids),
        "point_estimates_reproduce_saved_metrics": True,
        "interval_unit": "percentage points",
        "comparisons": {},
        "limits": [
            "Development split contains only 16 papers; intervals are not performance guarantees.",
            "These intervals condition on this trained model and local scoring protocol.",
            "They exclude annotation uncertainty, official-scorer differences and model-selection uncertainty.",
        ],
    }
    for metric in points[names[0]]:
        result["comparisons"][metric] = {
            **{
                name: {
                    "point": points[name][metric] * 100,
                    "ci95": [quantile(samples[name][metric], p) * 100 for p in [0.025, 0.975]],
                }
                for name in names
            },
            "training_aliases_minus_ontology": {
                "point": (points[names[1]][metric] - points[names[0]][metric]) * 100,
                "ci95": [quantile(samples["difference"][metric], p) * 100 for p in [0.025, 0.975]],
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.resamples, args.seed)
    result["analysis_code_sha256"] = pp.digest(__file__)
    pp.write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
