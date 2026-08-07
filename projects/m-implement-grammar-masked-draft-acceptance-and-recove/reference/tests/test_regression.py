import sys
import numpy as np

sys.path.insert(0, ".")
from speculative.acceptance import compute_accepted_length
from speculative.metrics import measure_acceptance_loss
from speculative.diagnostics import diagnose_collapse


def test_grammar_mask_is_enforced():
    draft_tokens = [1]
    target_probs = [np.array([0.1, 0.9])]
    draft_probs = [np.array([0.5, 0.5])]
    grammar_masks = [np.array([True, False])]
    random_samples = [0.0]

    got = compute_accepted_length(draft_tokens, target_probs, draft_probs, grammar_masks, random_samples)
    assert got == 0, "Grammar mask should reject the token"


def test_measure_loss():
    draft_tokens = [[1]]
    target_probs = [[np.array([0.1, 0.9])]]
    draft_probs = [[np.array([0.5, 0.5])]]
    grammar_masks = [[np.array([True, False])]]
    random_samples = [[0.0]]
    loss = measure_acceptance_loss(draft_tokens, target_probs, draft_probs, grammar_masks, random_samples)
    assert abs(loss - 1.0) < 1e-5


def test_diagnose():
    res = diagnose_collapse({"loss": 0.8}, {"loss": 0.1})
    assert res == "run_a"
