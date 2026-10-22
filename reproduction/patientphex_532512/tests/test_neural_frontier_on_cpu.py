"""Regressions for exact offsets, isolation, evidence identity, and probability joins."""

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest
from test_patientphex_on_cpu import document, entity

ROOT = Path(__file__).resolve().parents[1]
ANEMIA = "HP:0001903"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return tuple(importlib.import_module(n) for n in ("neural_frontier", "enhanced", "patientphex"))


def test_bio_supervision_ignores_partial_and_non_token_aligned_spans(modules):
    nf, _, _ = modules
    offsets = [(0, 0), (0, 3), (4, 7), (8, 12), (0, 0)]
    labels, counts = nf.bio_labels(offsets, [(0, 7), (8, 15)])
    assert labels == [-100, 1, 2, -100, -100]
    assert counts["complete_entity_instances"] == 1
    assert counts["unaligned_or_truncated"] == 1


def test_bio_supervision_deduplicates_and_records_overlaps(modules):
    nf, _, _ = modules
    labels, counts = nf.bio_labels([(0, 0), (0, 3), (4, 7), (0, 0)], [(0, 7), (0, 7), (4, 7)])
    assert labels == [-100, 1, 2, -100]
    assert counts["complete_entity_instances"] == 1
    assert counts["overlapping_annotation"] == 1


def test_bio_decoder_preserves_unicode_offsets_and_rejects_orphan_continuation(modules):
    nf, _, _ = modules
    offsets = [(0, 0), (0, 1), (2, 8), (8, 9), (0, 0)]
    probabilities = np.array([[1, 0, 0], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [1, 0, 0], [1, 0, 0]])
    spans = nf.decode_bio(offsets, probabilities, "İ anemia.")
    assert [(r["start"], r["end"]) for r in spans] == [(2, 8)]


def test_bio_decoder_rejects_truncated_entity_and_word_fragment(modules):
    nf, _, _ = modules
    probs = np.array([[1, 0, 0], [0.1, 0.8, 0.1], [1, 0, 0]])
    assert nf.decode_bio([(0, 0), (0, 6), (0, 0)], probs, "anemia and pain") == []
    assert nf.decode_bio([(0, 0), (2, 6), (6, 7)], probs, "anemia.") == []


def test_neural_model_rejects_duplicate_document_content(modules):
    nf, _, _ = modules
    fitting = document("P1 had anemia.", key="fit")
    different_id = copy.deepcopy(fitting)
    different_id["pmc_id"], different_id["pmid"] = "other", "other"
    with pytest.raises(ValueError, match="training/inference overlap"):
        nf.check_disjoint({"training_documents": nf.fingerprints([fitting])}, [different_id])


def test_merge_keeps_original_and_allows_only_containment_replacement(modules):
    nf, base, _ = modules
    doc = document("P1 had severe anemia and pain.")
    old = {
        "pmc_id": doc["pmc_id"],
        "pmid": doc["pmid"],
        "entities": [entity(doc, "anemia", ANEMIA)],
        "association": doc["association"],
    }
    span = entity(doc, "severe anemia", ANEMIA)
    linked = [{**span, "pmc_id": doc["pmc_id"], "exact_identifier": ANEMIA, "hits": []}]
    kept = nf.merge_entities(base, [doc], [old], linked, {"mode": "augment"})
    assert [e["text"] for e in kept[0]["entities"]] == ["anemia"]
    replaced = nf.merge_entities(base, [doc], [old], linked, {"mode": "replace"})
    assert [e["text"] for e in replaced[0]["entities"]] == ["severe anemia"]
    assert old["entities"][0]["text"] == "anemia"


def test_semantic_merge_rejects_ambiguous_concepts_and_invalid_text(modules):
    nf, base, _ = modules
    doc = document("P1 had anemia.")
    output = {"pmc_id": doc["pmc_id"], "pmid": doc["pmid"], "entities": [], "association": doc["association"]}
    row = {
        **entity(doc, "anemia", ANEMIA),
        "pmc_id": doc["pmc_id"],
        "exact_identifier": None,
        "hits": [{"identifier": ANEMIA, "score": 0.90}, {"identifier": "HP:0001156", "score": 0.89}],
    }
    settings = {"mode": "augment", "similarity": 0.85, "margin": 0.03}
    assert nf.merge_entities(base, [doc], [output], [row], settings)[0]["entities"] == []
    row["exact_identifier"], row["text"] = ANEMIA, "hallucinated"
    with pytest.raises(ValueError, match="span text mismatch"):
        nf.merge_entities(base, [doc], [output], [row], settings)


def test_retrieval_runner_up_is_a_distinct_concept(modules):
    nf, _, _ = modules
    retriever = nf.OntologyRetriever.__new__(nf.OntologyRetriever)
    retriever.pairs = [(f"synonym {i}", ANEMIA) for i in range(40)] + [("short fingers", "HP:0001156")]
    retriever.matrix = np.array([[1.0, 0.0]] * 40 + [[0.8, 0.6]], dtype=np.float32)
    retriever.phrases = {}
    retriever.encode = lambda texts: np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)
    hits = retriever.retrieve(["anemia"])["anemia"]
    assert [h["identifier"] for h in hits] == [ANEMIA, "HP:0001156"]
    assert hits[1]["score"] == pytest.approx(0.8)


def test_relation_evidence_ignores_source_gold_and_marks_other_patient(modules):
    nf, base, _ = modules
    doc = document("P1 had anemia. P2 was healthy.")
    output = {
        "pmc_id": doc["pmc_id"],
        "pmid": doc["pmid"],
        "entities": [entity(doc, "anemia", ANEMIA)],
        "association": doc["association"],
    }
    ontology = {"terms": {ANEMIA: {"name": "Anemia"}}}
    expected = nf.relation_texts(base, [doc], [output], ontology)
    poisoned = copy.deepcopy(doc)
    poisoned["entities"], poisoned["association"] = [{"invalid": True}], [{"invalid": True}]
    assert nf.relation_texts(base, [poisoned], [output], ontology) == expected
    assert len(expected) == 2
    assert "[TARGET_PATIENT]" in expected[0]["context"]
    assert "[OTHER_PATIENT]" in expected[0]["context"]
    assert "[PHENOTYPE]" in expected[0]["context"]


def test_blend_joins_by_identity_and_rejects_missing_candidates(modules):
    nf, _, _ = modules
    classical = [{"pmc_id": "d", "patient_id": "p", "concept": ANEMIA, "probability": 0.2}]
    neural = [{**classical[0], "probability": 0.8, "evidence": []}]
    assert nf.blend_scores(classical, neural, 0.5)[0]["probability"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="candidate mismatch"):
        nf.blend_scores(classical, [], 0.5)
    with pytest.raises(ValueError, match="duplicate neural"):
        nf.blend_scores(classical, neural * 2, 0.5)
