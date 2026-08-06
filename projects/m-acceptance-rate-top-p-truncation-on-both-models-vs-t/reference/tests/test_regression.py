import numpy as np
from spec.sampling import sample_residual


def test_acceptance_probability_residual_masking():
    np.random.seed(42)
    p_logits = np.array([3.0, 4.0, 0.1])
    q_logits = np.array([0.1, 2.0, 1.9])
    tokens = [
        sample_residual(
            p_logits,
            q_logits,
            temperature=1.0,
            top_p_target=1.0,
            top_p_draft=0.9,
        )
        for _ in range(50)
    ]
    assert 0 in tokens
