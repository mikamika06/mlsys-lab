import numpy as np
from grammar.transition import build_transition_matrix
from grammar.ceiling import compute_acceptance_ceiling


def test_ceiling_properties():
    spec = {
        "states": ["S0", "S1"],
        "vocab_size": 4,
        "transitions": [
            ("S0", 0, "S1"),
            ("S0", 1, "S1"),
            ("S1", 2, "S0"),
            ("S1", 3, "S1"),
        ],
    }
    mat = build_transition_matrix(spec)
    ceiling = compute_acceptance_ceiling(mat, 5)
    assert 0.0 <= ceiling <= 5.0

    spec_terminal = {
        "states": ["S0"],
        "vocab_size": 2,
        "transitions": [
            ("S0", 0, "S0"),
        ],
    }
    mat_term = build_transition_matrix(spec_terminal)
    ceiling_term = compute_acceptance_ceiling(mat_term, 3)
    assert ceiling_term > 0.0
