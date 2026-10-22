"""Run the frozen PatientPheX neural experiment, with resumable completed stages."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import lexical_recall as recall
import neural_frontier as nf
import patient_sets
import patientphex as pp
from run_optimization import bootstrap_difference, compact_rows, write_variant
from run_set_optimization import frozen_base


def read(path):
    return json.loads(Path(path).read_text())


def save_model(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        pickle.dump(model, handle, protocol=5)


def load_model(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def prepare(args):
    run, prior = Path(args.output_dir), Path(args.previous_run)
    if (run / "config.json").exists():
        config = read(run / "config.json")
        for name, digest in config["source_sha256"].items():
            pp.require(pp.digest(run / "source" / name) == digest, f"frozen source changed: {name}")
        return config
    pp.require(not run.exists(), f"incomplete existing run directory: {run}")
    base_config = read(prior / "config.json")
    source = run / "source"
    source.mkdir(parents=True)
    for name in ("patientphex.py", "enhanced.py", "run_baselines.py", "analyze_baselines.py"):
        shutil.copyfile(prior / "source" / name, source / name)
    for name in (
        "neural_frontier.py",
        "run_neural_frontier.py",
        "predict_neural_bundle.py",
        "lexical_recall.py",
        "run_optimization.py",
        "run_set_optimization.py",
        "patient_sets.py",
    ):
        shutil.copyfile(Path(__file__).with_name(name), source / name)
    shutil.copyfile(Path(__file__).with_name("FRONTIER_EXECUTION_PROTOCOL.md"), run / "protocol.md")
    packages = {
        **base_config["packages"],
        **{name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "safetensors")},
    }
    model_root = Path(args.model_root).resolve()
    manifests = {name: read(model_root / f"{name}.manifest.json") for name in ("biolinkbert_base", "sapbert")}
    for name, manifest in manifests.items():
        for filename, info in manifest["files"].items():
            pp.require(
                pp.digest(model_root / name / filename) == info["sha256"],
                f"pretrained hash mismatch: {name}/{filename}",
            )
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "previous_run": str(prior.resolve()),
        "baseline_run": base_config["baseline_run"],
        "inputs": base_config["inputs"],
        "folds": base_config["folds"],
        "lexical_config": base_config["lexical_config"],
        "recall_config": read(prior / "selection.json")["selected"]["recall_config"],
        "model_root": str(model_root),
        "artifact_root": str(Path(args.artifact_root).resolve()),
        "device_requested": args.device,
        "model_manifests": manifests,
        "ner_settings": nf.NER_SETTINGS,
        "relation_settings": nf.RELATION_SETTINGS,
        "entity_grid": nf.entity_grid(),
        "weights": [0, 0.25, 0.5, 0.75, 1],
        "thresholds": [0.2, 0.35, 0.5, 0.65],
        "packages": packages,
        "source_sha256": {p.name: pp.digest(p) for p in source.iterdir()},
        "protocol_sha256": pp.digest(run / "protocol.md"),
    }
    for name, item in config["inputs"].items():
        pp.require(pp.digest(item["path"]) == item["sha256"], f"input changed: {name}")
    pp.write_json(run / "config.json", config)
    return config


class Experiment:
    def __init__(self, args):
        self.run = Path(args.output_dir)
        self.config = prepare(args)
        self.device = args.device
        self.prior = Path(self.config["previous_run"])
        self.base = frozen_base(self.prior)
        self.inputs = self.config["inputs"]
        self.ontology = pp.load_ontology(self.inputs["hpo"]["path"])
        self.train = pp.read_jsonl(Path(self.config["baseline_run"]) / "split/train.jsonl")
        indexed = pp.index_documents(self.train, source=True)
        self.folds = [[indexed[key] for key in ids] for ids in self.config["folds"]]
        self.artifacts = Path(self.config["artifact_root"])
        self.models = Path(self.config["model_root"])

    def fitting(self, index):
        return [d for i, fold in enumerate(self.folds) if i != index for d in fold]

    def prior_fold(self, index):
        return self.prior / f"inner/all_longest/fold{index}"

    def lexical_fit(self, documents):
        return recall.fit_lexical(
            self.base,
            self.ontology,
            documents,
            self.config["lexical_config"],
            self.config["recall_config"],
            self.inputs["hpo"]["path"],
        )

    def checkpoint(self, scope, task):
        return self.artifacts / scope / task

    def completed_checkpoint(self, path):
        if (path / "training_metadata.json").exists():
            metadata = read(path / "training_metadata.json")
            pp.require(metadata["state"] == "complete", "incomplete checkpoint")
            pp.require(pp.digest(path / "model.safetensors") == metadata["model_sha256"], "checkpoint hash mismatch")
            return True
        return False

    def neural_spans(self, scope, task_name, documents, checkpoint, overlap=False):
        folder = self.run / scope
        path = folder / f"{task_name}.spans.jsonl"
        if path.exists():
            return pp.read_jsonl(path)
        rows, audit = nf.predict_ner(documents, checkpoint, self.device, allow_training_overlap=overlap)
        pp.write_json(path, rows, True)
        pp.write_json(folder / f"{task_name}.span_audit.json", audit)
        return rows

    def link(self, scope, name, spans, lexical, retriever):
        path = self.run / scope / f"{name}.linked.jsonl"
        if path.exists():
            return pp.read_jsonl(path)
        rows = nf.link_spans(spans, lexical, retriever)
        pp.write_json(path, rows, True)
        return rows

    def retriever(self):
        return nf.OntologyRetriever(
            self.models / "sapbert", self.ontology, self.artifacts / "ontology_index", self.device
        )

    def inner_ner(self):
        for index, validation in enumerate(self.folds):
            scope = f"inner/fold{index}"
            fitting = self.fitting(index)
            checkpoint = self.checkpoint(scope, "ner")
            if not self.completed_checkpoint(checkpoint):
                nf.train_ner(fitting, self.models / "biolinkbert_base", checkpoint, self.device)
            self.neural_spans(scope, "validation", validation, checkpoint)
            self.neural_spans(scope, "fitting", fitting, checkpoint, overlap=True)
            nf.emit(stage="ner_fold_complete", fold=index)
        retriever = self.retriever()
        for index, validation in enumerate(self.folds):
            scope = f"inner/fold{index}"
            lexical = read(self.prior_fold(index) / "lexical.json")
            for name in ("validation", "fitting"):
                spans = pp.read_jsonl(self.run / scope / f"{name}.spans.jsonl")
                self.link(scope, name, spans, lexical, retriever)
        retriever.close()

    def select_entities(self):
        path = self.run / "entity_selection.json"
        if path.exists():
            return read(path)["selected"]
        candidates = []
        for settings in self.config["entity_grid"]:
            pooled = []
            for index, validation in enumerate(self.folds):
                lexical = pp.read_jsonl(self.prior_fold(index) / "entities.jsonl")
                linked = pp.read_jsonl(self.run / f"inner/fold{index}/validation.linked.jsonl")
                pooled.extend(nf.merge_entities(self.base, validation, lexical, linked, settings))
            metrics = pp.evaluate(self.train, pooled)
            candidates.append(
                {
                    "settings": settings,
                    "metrics": metrics,
                    "selection_score": (metrics["mention"]["f1"] + metrics["document"]["f1"]) / 2,
                }
            )
        selected = max(candidates, key=lambda r: r["selection_score"])
        pp.write_json(path, {"selected": selected, "candidates": candidates, "outer_labels_read": False})
        nf.emit(stage="entity_selected", settings=selected["settings"], score=selected["selection_score"])
        return selected

    def inner_predictions(self, index, name, settings):
        scope = f"inner/fold{index}"
        output = self.run / scope / f"{name}.entities.jsonl"
        if output.exists():
            return pp.read_jsonl(output)
        documents = self.folds[index] if name == "validation" else self.fitting(index)
        lexical = read(self.prior_fold(index) / "lexical.json")
        old, _ = recall.predict_lexical(self.base, lexical, documents, allow_training_overlap=name == "fitting")
        linked = pp.read_jsonl(self.run / scope / f"{name}.linked.jsonl")
        predictions = nf.merge_entities(self.base, documents, old, linked, settings)
        pp.validate_predictions(documents, predictions, self.ontology)
        pp.write_json(output, predictions, True)
        return predictions

    def inner_relations(self):
        settings = self.select_entities()["settings"]
        for index, validation in enumerate(self.folds):
            scope = f"inner/fold{index}"
            folder = self.run / scope
            fitting = self.fitting(index)
            fitted_entities = self.inner_predictions(index, "fitting", settings)
            entities = self.inner_predictions(index, "validation", settings)
            checkpoint = self.checkpoint(scope, "relation")
            if not self.completed_checkpoint(checkpoint):
                nf.train_relation(
                    self.base,
                    fitting,
                    fitted_entities,
                    self.ontology,
                    self.models / "biolinkbert_base",
                    checkpoint,
                    self.device,
                )
            if not (folder / "neural_scores.jsonl").exists():
                scores = nf.predict_relation(self.base, validation, entities, self.ontology, checkpoint, self.device)
                pp.write_json(folder / "neural_scores.jsonl", scores, True)
            if not (folder / "classical_scores.jsonl").exists():
                model_path = folder / "relation.pkl"
                if not model_path.exists():
                    model = self.base.fit_relations(fitting, fitted_entities, self.ontology)
                    save_model(model_path, model)
                model = load_model(model_path)
                scores = self.base.score_relations(model, validation, entities)
                pp.write_json(folder / "classical_scores.jsonl", compact_rows(scores), True)
            nf.emit(stage="relation_fold_complete", fold=index)

    def select_complete(self):
        path = self.run / "selection.json"
        if path.exists():
            return read(path)["selected"]
        pooled_entities, classical, neural, old = [], [], [], []
        for index, validation in enumerate(self.folds):
            folder = self.run / f"inner/fold{index}"
            pooled_entities.extend(pp.read_jsonl(folder / "validation.entities.jsonl"))
            classical.extend(pp.read_jsonl(folder / "classical_scores.jsonl"))
            neural.extend(pp.read_jsonl(folder / "neural_scores.jsonl"))
            old.extend(
                self.base.apply_relations(
                    validation,
                    pp.read_jsonl(self.prior_fold(index) / "entities.jsonl"),
                    pp.read_jsonl(self.prior_fold(index) / "scores.jsonl"),
                    0.2,
                )
            )
        candidates = [{"name": "baseline", "weight": 0, "threshold": 0.2, "metrics": pp.evaluate(self.train, old)}]
        for weight in self.config["weights"]:
            scores = nf.blend_scores(classical, neural, weight)
            for threshold in self.config["thresholds"]:
                predictions = self.base.apply_relations(self.train, pooled_entities, scores, threshold)
                candidates.append(
                    {
                        "name": "neural_frontier",
                        "weight": weight,
                        "threshold": threshold,
                        "metrics": pp.evaluate(self.train, predictions),
                    }
                )
        selected = max(candidates, key=lambda r: r["metrics"]["score"])
        pp.write_json(
            path,
            {
                "selected": selected,
                "candidates": candidates,
                "outer_labels_read": False,
                "selected_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        nf.emit(
            stage="complete_selected",
            name=selected["name"],
            weight=selected["weight"],
            threshold=selected["threshold"],
            score=selected["metrics"]["score"],
        )
        return selected

    def fit_scope(self, scope, training, targets):
        settings = self.select_entities()["settings"]
        folder = self.run / scope
        folder.mkdir(parents=True, exist_ok=True)
        lexical_path = folder / "lexical.json"
        if not lexical_path.exists():
            pp.write_json(lexical_path, self.lexical_fit(training))
        lexical = read(lexical_path)
        ner = self.checkpoint(scope, "ner")
        if not self.completed_checkpoint(ner):
            nf.train_ner(training, self.models / "biolinkbert_base", ner, self.device)
        span_sets = {"fitting": self.neural_spans(scope, "fitting", training, ner, True)}
        for name, documents in targets.items():
            span_sets[name] = self.neural_spans(scope, name, documents, ner)
        retriever = self.retriever()
        entities = {}
        for name, documents in {"fitting": training, **targets}.items():
            output = folder / f"{name}.entities.jsonl"
            if output.exists():
                entities[name] = pp.read_jsonl(output)
                continue
            linked = self.link(scope, name, span_sets[name], lexical, retriever)
            previous, _ = recall.predict_lexical(
                self.base, lexical, documents, allow_training_overlap=name == "fitting"
            )
            entities[name] = nf.merge_entities(self.base, documents, previous, linked, settings)
            pp.write_json(output, entities[name], True)
        retriever.close()
        relation = self.checkpoint(scope, "relation")
        if not self.completed_checkpoint(relation):
            nf.train_relation(
                self.base,
                training,
                entities["fitting"],
                self.ontology,
                self.models / "biolinkbert_base",
                relation,
                self.device,
            )
        model_path = folder / "relation.pkl"
        if not model_path.exists():
            save_model(model_path, self.base.fit_relations(training, entities["fitting"], self.ontology))
        classifier = load_model(model_path)
        for name, documents in targets.items():
            neural_path = folder / f"{name}.neural_scores.jsonl"
            if not neural_path.exists():
                pp.write_json(
                    neural_path,
                    nf.predict_relation(self.base, documents, entities[name], self.ontology, relation, self.device),
                    True,
                )
            classical_path = folder / f"{name}.classical_scores.jsonl"
            if not classical_path.exists():
                pp.write_json(
                    classical_path, compact_rows(self.base.score_relations(classifier, documents, entities[name])), True
                )

    def outer(self):
        selected = self.select_complete()
        decision_path = self.run / "outer/decision.json"
        if decision_path.exists():
            return read(decision_path)
        dev = pp.read_jsonl(Path(self.config["baseline_run"]) / "split/dev.gold.jsonl")
        old = pp.read_jsonl(self.prior / "outer/selected/predictions.jsonl")
        variants = [("baseline", old, [])]
        if selected["name"] != "baseline":
            self.fit_scope("outer", self.train, {"development": dev})
            folder = self.run / "outer"
            entities = pp.read_jsonl(folder / "development.entities.jsonl")
            scores = nf.blend_scores(
                pp.read_jsonl(folder / "development.classical_scores.jsonl"),
                pp.read_jsonl(folder / "development.neural_scores.jsonl"),
                selected["weight"],
            )
            predictions = self.base.apply_relations(dev, entities, scores, selected["threshold"])
            old_relation = load_model(self.prior / "outer/relation.pkl")
            old_scores = self.base.score_relations(old_relation, dev, entities)
            ablation = self.base.apply_relations(dev, entities, old_scores, 0.2)
            variants.extend(
                [("selected", predictions, scores), ("new_entities_old_relation", ablation, compact_rows(old_scores))]
            )
        else:
            variants.append(("selected", old, []))
        metrics, diagnoses = {}, {}
        for name, predictions, evidence in variants:
            pp.validate_predictions(dev, predictions, self.ontology)
            metric, diagnosis = write_variant(self.run, f"outer/{name}", dev, predictions, evidence)
            metrics[name] = {"metrics": metric, "patient_groups": patient_sets.grouped_patient_metrics(diagnosis)}
            diagnoses[name] = diagnosis
        lhs, rhs = (metrics[name]["metrics"] for name in ("baseline", "selected"))
        adopt = rhs["score"] > lhs["score"] + 1e-12 and rhs["patient_macro"]["f1"] >= lhs["patient_macro"]["f1"] - 1e-12
        result = {
            "adopt": adopt,
            "variants": metrics,
            "paired_difference": bootstrap_difference(
                diagnoses["baseline"]["documents"], diagnoses["selected"]["documents"]
            ),
        }
        pp.write_json(decision_path, result)
        nf.emit(stage="outer_complete", adopt=adopt, score=rhs["score"], macro=rhs["patient_macro"]["f1"])
        return result

    def full(self):
        decision = self.outer()
        exports = {}
        if decision["adopt"]:
            training = pp.read_jsonl(self.inputs["train"]["path"])
            targets = {name: pp.read_jsonl(self.inputs[name]["path"]) for name in ("a", "b")}
            self.fit_scope("full", training, targets)
            folder = self.run / "full"
            selected = self.select_complete()
            for name, documents in targets.items():
                path = folder / f"{name}.predictions.jsonl"
                if not path.exists():
                    entities = pp.read_jsonl(folder / f"{name}.entities.jsonl")
                    scores = nf.blend_scores(
                        pp.read_jsonl(folder / f"{name}.classical_scores.jsonl"),
                        pp.read_jsonl(folder / f"{name}.neural_scores.jsonl"),
                        selected["weight"],
                    )
                    predictions = self.base.apply_relations(documents, entities, scores, selected["threshold"])
                    pp.write_json(path, predictions, True)
                    pp.write_json(folder / f"{name}.evidence.jsonl", scores, True)
                validation = pp.validate_predictions(documents, pp.read_jsonl(path), self.ontology)
                exports[name] = {"path": str(path.resolve()), "sha256": pp.digest(path), "validation": validation}
            self.bundle()
        summary = {
            "state": "complete",
            "adopt": decision["adopt"],
            "selection": self.select_complete(),
            "outer": decision,
            "exports": exports,
            "submission": None,
        }
        if not (self.run / "summary.json").exists():
            pp.write_json(self.run / "summary.json", summary)
        return summary

    def bundle(self):
        folder = self.run / "full"
        if (folder / "bundle_manifest.json").exists():
            return
        inference = {
            "entity_settings": self.select_entities()["settings"],
            "relation_settings": {k: self.select_complete()[k] for k in ("weight", "threshold")},
            "ner_checkpoint": str(self.checkpoint("full", "ner")),
            "relation_checkpoint": str(self.checkpoint("full", "relation")),
            "sapbert": str(self.models / "sapbert"),
            "ontology_cache": str(self.artifacts / "ontology_index"),
        }
        pp.write_json(folder / "inference_config.json", inference)
        external = {}
        for directory in (
            self.checkpoint("full", "ner"),
            self.checkpoint("full", "relation"),
            self.models / "sapbert",
            self.artifacts / "ontology_index",
        ):
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    external[str(path.resolve())] = pp.digest(path)
        manifest = {
            "method": "neural_frontier_v1",
            "source_sha256": self.config["source_sha256"],
            "packages": self.config["packages"],
            "model_sha256": {
                name: pp.digest(folder / name) for name in ("lexical.json", "relation.pkl", "inference_config.json")
            },
            "external_sha256": external,
            "hpo_sha256": self.ontology["sha256"],
            "training_sha256": self.inputs["train"]["sha256"],
        }
        pp.write_json(folder / "bundle_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", default="reproduction/patientphex_532512/runs/recall_optimization_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stage", choices=("prepare", "ner", "entities", "relations", "select", "outer", "full", "all"), default="all"
    )
    args = parser.parse_args()
    started = time.monotonic()
    experiment = Experiment(args)
    stages = {
        "ner": experiment.inner_ner,
        "entities": experiment.select_entities,
        "relations": experiment.inner_relations,
        "select": experiment.select_complete,
        "outer": experiment.outer,
        "full": experiment.full,
    }
    if args.stage == "all":
        for function in stages.values():
            function()
    elif args.stage != "prepare":
        stages[args.stage]()
    nf.emit(stage="command_complete", command=args.stage, seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
