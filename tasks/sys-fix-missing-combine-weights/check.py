import numpy as np

def _oracle_combine(expert_outputs, gate_weights):
    """Reference computation: weighted sum over experts."""
    return np.sum(gate_weights[:, np.newaxis] * expert_outputs, axis=0)

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(20240101)

    # Hand-crafted edge / diagnostic cases
    cases = [
        # Uniform weights — unweighted mean happens to match (does NOT catch the bug alone)
        (np.ones((4, 8)), np.array([0.25, 0.25, 0.25, 0.25])),
        # All weight on expert 0
        (np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]]),
         np.array([1.0, 0.0, 0.0])),
        # All weight on the last expert
        (np.array([[10.0, 20.0],
                    [30.0, 40.0]]),
         np.array([0.0, 1.0])),
        # Sparse-ish weights, divergent experts
        (np.array([[0.0, 1.0],
                    [5.0, 5.0],
                    [-3.0, 2.0]]),
         np.array([0.7, 0.2, 0.1])),
    ]

    # Random cases — varying sizes and distributions
    for _ in range(20):
        n_experts = rng.randint(2, 12)
        d = rng.randint(2, 64)
        expert_outputs = rng.randn(n_experts, d)
        logits = rng.randn(n_experts)
        gate_weights = np.exp(logits) / np.sum(np.exp(logits))
        cases.append((expert_outputs, gate_weights))

    worst_rel_err = 0.0
    for expert_outputs, gate_weights in cases:
        try:
            got = np.asarray(sol.moe_combine(expert_outputs, gate_weights),
                             dtype=np.float64)
        except Exception:
            return {"rel_err": 1.0}
        ref = _oracle_combine(expert_outputs, gate_weights)
        norm_ref = np.linalg.norm(ref)
        if norm_ref < 1e-15:
            # Both should be (near) zero; use absolute check
            err = float(np.linalg.norm(got - ref))
        else:
            err = float(np.linalg.norm(got - ref) / norm_ref)
        if err > worst_rel_err:
            worst_rel_err = err

    return {"rel_err": worst_rel_err}
