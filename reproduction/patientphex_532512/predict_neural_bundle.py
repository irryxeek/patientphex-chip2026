"""Verify and run an immutable neural PatientPheX bundle in a fresh process."""

import argparse
import importlib
import importlib.metadata
import json
import pickle
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--hpo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    directory = Path(args.model_dir).resolve()
    manifest = json.loads((directory / "bundle_manifest.json").read_text())
    source = directory.parent / "source"
    import hashlib

    def digest(path):
        value = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1048576), b""):
                value.update(block)
        return value.hexdigest()

    for relative, expected in manifest["source_sha256"].items():
        if digest(source / relative) != expected:
            raise ValueError(f"frozen source hash mismatch: {relative}")
    for relative, expected in manifest["model_sha256"].items():
        if digest(directory / relative) != expected:
            raise ValueError(f"model hash mismatch: {relative}")
    for path, expected in manifest["external_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"external model hash mismatch: {path}")
    for name, expected in manifest["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"package mismatch: {name}")
    sys.path.insert(0, str(source))
    pp, nf, recall, base = [
        importlib.import_module(name) for name in ("patientphex", "neural_frontier", "lexical_recall", "enhanced")
    ]
    pp.require(digest(args.hpo) == manifest["hpo_sha256"], "HPO mismatch")
    outputs = [Path(args.output).resolve(), Path(args.evidence).resolve()]
    pp.require(len(set(outputs)) == 2 and not any(p.exists() for p in outputs), "output exists or paths collide")
    config = json.loads((directory / "inference_config.json").read_text())
    lexical = json.loads((directory / "lexical.json").read_text())
    documents, ontology = pp.read_jsonl(args.input), pp.load_ontology(args.hpo)
    entities, _ = recall.predict_lexical(base, lexical, documents)
    spans, _ = nf.predict_ner(documents, config["ner_checkpoint"], args.device)
    retriever = nf.OntologyRetriever(config["sapbert"], ontology, config["ontology_cache"], args.device)
    linked = nf.link_spans(spans, lexical, retriever)
    retriever.close()
    entities = nf.merge_entities(base, documents, entities, linked, config["entity_settings"])
    with (directory / "relation.pkl").open("rb") as handle:
        relation = pickle.load(handle)
    classical = base.score_relations(relation, documents, entities)
    neural = nf.predict_relation(base, documents, entities, ontology, config["relation_checkpoint"], args.device)
    settings = config["relation_settings"]
    scores = nf.blend_scores(classical, neural, settings["weight"])
    predictions = base.apply_relations(documents, entities, scores, settings["threshold"])
    validation = pp.validate_predictions(documents, predictions, ontology)
    pp.write_json(args.output, predictions, True)
    pp.write_json(args.evidence, scores, True)
    print(json.dumps({"validation": validation, "sha256": digest(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
