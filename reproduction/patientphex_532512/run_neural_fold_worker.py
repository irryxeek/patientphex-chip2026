"""Schedule independent folds on an assigned GPU using the run's frozen implementation."""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--stage", choices=("ner", "relations"), required=True)
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = Path(args.run).resolve()
    config = json.loads((run / "config.json").read_text())
    if len(set(args.folds)) != len(args.folds) or any(i not in range(len(config["folds"])) for i in args.folds):
        raise ValueError("invalid or duplicated fold indices")
    sys.path.insert(0, str(run / "source"))
    import run_neural_frontier as frozen

    experiment = frozen.Experiment(
        SimpleNamespace(
            output_dir=str(run),
            previous_run=config["previous_run"],
            artifact_root=config["artifact_root"],
            model_root=config["model_root"],
            device=args.device,
        )
    )
    nf, pp = frozen.nf, frozen.pp
    started = time.monotonic()
    for index in args.folds:
        scope = f"inner/fold{index}"
        folder = run / scope
        validation, fitting = experiment.folds[index], experiment.fitting(index)
        if args.stage == "ner":
            checkpoint = experiment.checkpoint(scope, "ner")
            if not experiment.completed_checkpoint(checkpoint):
                nf.train_ner(fitting, experiment.models / "biolinkbert_base", checkpoint, args.device)
            spans = {
                "validation": experiment.neural_spans(scope, "validation", validation, checkpoint),
                "fitting": experiment.neural_spans(scope, "fitting", fitting, checkpoint, overlap=True),
            }
            lexical = frozen.read(experiment.prior_fold(index) / "lexical.json")
            retriever = experiment.retriever()
            for name, rows in spans.items():
                experiment.link(scope, name, rows, lexical, retriever)
            retriever.close()
        else:
            settings = frozen.read(run / "entity_selection.json")["selected"]["settings"]
            fitted = experiment.inner_predictions(index, "fitting", settings)
            entities = experiment.inner_predictions(index, "validation", settings)
            checkpoint = experiment.checkpoint(scope, "relation")
            if not experiment.completed_checkpoint(checkpoint):
                nf.train_relation(
                    experiment.base,
                    fitting,
                    fitted,
                    experiment.ontology,
                    experiment.models / "biolinkbert_base",
                    checkpoint,
                    args.device,
                )
            if not (folder / "neural_scores.jsonl").exists():
                scores = nf.predict_relation(
                    experiment.base, validation, entities, experiment.ontology, checkpoint, args.device
                )
                pp.write_json(folder / "neural_scores.jsonl", scores, True)
            if not (folder / "classical_scores.jsonl").exists():
                model_path = folder / "relation.pkl"
                if not model_path.exists():
                    model = experiment.base.fit_relations(fitting, fitted, experiment.ontology)
                    frozen.save_model(model_path, model)
                model = frozen.load_model(model_path)
                scores = experiment.base.score_relations(model, validation, entities)
                pp.write_json(folder / "classical_scores.jsonl", frozen.compact_rows(scores), True)
        nf.emit(stage=args.stage + "_fold_complete", fold=index, seconds=time.monotonic() - started)
    nf.emit(stage="worker_complete", task=args.stage, folds=args.folds, seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
