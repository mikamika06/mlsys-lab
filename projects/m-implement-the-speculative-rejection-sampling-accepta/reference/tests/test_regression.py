import numpy as np
from speculative.accept import evaluate_draft

def test_identical_distributions_always_accept():
    target_p = np.array([[0.2, 0.8]])
    draft_p = np.array([[0.2, 0.8]])
    tokens = np.array([0])
    u = np.array([0.5])

    n, residual = evaluate_draft(target_p, draft_p, tokens, u)
    assert n == 1
    assert residual is None
