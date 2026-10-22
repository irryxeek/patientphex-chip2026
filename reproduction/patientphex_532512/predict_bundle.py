"""Run a locally fitted PatientPheX bundle using its frozen inference source."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import pickle
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    model_dir = Path(args.model_dir).resolve()
    manifest = json.loads((model_dir / "bundle_manifest.json").read_text())
    for filename, expected in manifest["model_sha256"].items():
        if sha256(model_dir / filename) != expected:
            raise ValueError(f"model hash mismatch: {filename}")
    source_dir = model_dir.parent / "source"
    for filename, expected in manifest["source_sha256"].items():
        if sha256(source_dir / filename) != expected:
            raise ValueError(f"frozen source hash mismatch: {filename}")
    for package, expected in manifest["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"package version mismatch: {package}; expected {expected}")
    sys.path.insert(0, str(source_dir))
    # This command is a fresh process. Do not reuse another run's imported implementation.
    for name in ("patientphex", "enhanced", "patient_sets", "lexical_recall"):
        if name in sys.modules:
            raise ValueError("run bundle inference in a fresh Python process")
    pp = importlib.import_module("patientphex")
    enhanced = importlib.import_module("enhanced")
    config = json.loads((model_dir / "inference_config.json").read_text())
    lexical = json.loads((model_dir / config["lexical_file"]).read_text())
    ontology = pp.load_ontology(args.hpo)
    pp.require(ontology["sha256"] == lexical["hpo_sha256"], "bundle HPO hash mismatch")
    source = pp.read_jsonl(args.input)
    if config.get("lexical_backend") == "recall":
        recall = importlib.import_module("lexical_recall")
        outputs, evidence = recall.predict_lexical(enhanced, lexical, source)
    else:
        outputs, evidence = enhanced.predict_lexical(lexical, source)
    if config["relation"] in {"classifier", "patient_set_classifier"}:
        # Only the fitted local checkpoint whose hash was verified above is deserialized.
        with (model_dir / config["relation_file"]).open("rb") as handle:
            relation = pickle.load(handle)
        rows = enhanced.score_relations(relation, source, outputs)
        if config["relation"] == "patient_set_classifier":
            patient_sets = importlib.import_module("patient_sets")
            outputs, evidence = patient_sets.apply_policy(source, outputs, rows, config["set_policy"])
        else:
            outputs = enhanced.apply_relations(source, outputs, rows, config["threshold"])
            evidence = [{key: value for key, value in row.items() if key != "features"} for row in rows]
    validation = pp.validate_predictions(source, outputs, ontology)
    paths = [Path(path).resolve() for path in (args.output, args.evidence) if path]
    pp.require(len(paths) == len(set(paths)), "prediction/evidence output paths must differ")
    pp.require(all(not path.exists() for path in paths), "refusing to overwrite existing outputs")
    pp.write_json(args.output, outputs, True)
    if args.evidence:
        pp.write_json(args.evidence, evidence, True)
    result = {
        "validation": validation,
        "prediction_sha256": sha256(args.output),
        "input_sha256": sha256(args.input),
        "bundle_manifest_sha256": sha256(model_dir / "bundle_manifest.json"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--hpo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
