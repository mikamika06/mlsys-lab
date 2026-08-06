import numpy as np
from speculative.acceptance import compute_accepted_length
from speculative.metrics import measure_acceptance_loss
from speculative.diagnostics import diagnose_collapse


def test_acceptance_with_grammar():
    draft_tokens = [10, 20, 30]
    target_probs = [np.array([0.1, 0.9]), np.array([0.2, 0.8]), np.array([0.5, 0.5])]
    draft_probs = [np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    grammar_mask = [{10: True}, {20: False}, {30: True}]
    length = compute_accepted_length(draft_tokens, target_probs, draft_probs, grammar_mask)
    assert length == 0


def test_recovery_length():
    draft_tokens = [5, 6, 7]
    target_probs = [np.array([0.1, 0.9]), np.array([0.1, 0.9]), np.array([0.1, 0.9])]
    draft_probs = [np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    grammar_mask = [{5: True}, {6: True}, {7: True}]
    length = compute_accepted_length(draft_tokens, target_probs, draft_probs, grammar_mask)
    assert length >= 0


def test_diagnostic_flag():
    res = diagnose_collapse({"acceptance_loss": 0.1}, {"acceptance_loss": 0.5})
    assert res == "collapsed"
