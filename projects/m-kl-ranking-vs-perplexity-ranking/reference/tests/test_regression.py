import sys

sys.path.insert(0, ".")
import numpy as np
from evalrec.gate import evaluate_acceptance_gate
from evalrec.metrics import rank_quant_candidates


def test_per_category_threshold_rejection():
    candidate_metrics = {
        "cat_a": {"kl": 0.01, "ppl": 5.0},
        "cat_b": {"kl": 0.25, "ppl": 5.0},
    }
    thresholds = {
        "cat_a": {"max_kl": 0.20, "max_ppl": 10.0},
        "cat_b": {"max_kl": 0.20, "max_ppl": 10.0},
    }
    res = evaluate_acceptance_gate(candidate_metrics, thresholds)
    assert not res[
        "accepted"
    ], "Candidate with one failing category must be rejected"
    assert "cat_b" in res["failed_categories"]
    assert res["category_results"]["cat_b"]["passed"] is False
    assert res["category_results"]["cat_a"]["passed"] is True


def test_kl_ranking_order_independent_of_ppl():
    teacher_data = {
        "cat1": {
            "logits": np.array([[2.0, 0.0], [1.0, 1.0]]),
            "targets": np.array([0, 1]),
        }
    }
    c1_logits = np.array([[1.9, 0.1], [0.9, 1.1]])
    c2_logits = np.array([[10.0, -10.0], [-10.0, 10.0]])

    candidates_data = {
        "cand_low_kl": {"cat1": {"logits": c1_logits}},
        "cand_low_ppl": {"cat1": {"logits": c2_logits}},
    }

    res = rank_quant_candidates(teacher_data, candidates_data)
    cand_kl = res["candidates"]["cand_low_kl"]
    cand_ppl = res["candidates"]["cand_low_ppl"]

    assert cand_kl["kl_rank"] == 1, "cand_low_kl must have KL rank 1"
    assert cand_ppl["kl_rank"] == 2, "cand_low_ppl must have KL rank 2"
    assert cand_ppl["ppl_rank"] == 1, "cand_low_ppl must have PPL rank 1"
    assert res["rank_disagreement"] is True
