import sys
sys.path.insert(0, ".")
from speculative.sampling import accept_reject_prob

def test_acceptance_probability_bounds():
    for p_val in [0.1, 0.5, 0.9]:
        for q_val in [0.1, 0.5, 0.9]:
            prob = accept_reject_prob(p_val, q_val)
            assert 0.0 <= prob <= 1.0, f"acceptance probability {prob} out of bounds [0, 1]"

def test_acceptance_exact_values():
    assert accept_reject_prob(0.4, 0.2) == 1.0
    assert abs(accept_reject_prob(0.2, 0.4) - 0.5) < 1e-6
