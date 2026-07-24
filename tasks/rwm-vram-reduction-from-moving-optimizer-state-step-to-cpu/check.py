import numpy as np
from mlsys import scorers


def _oracle(n_params, param_bytes, grad_bytes, master_bytes, m_bytes, v_bytes, activation_bytes):
    n = int(n_params)
    total_before = n * (param_bytes + grad_bytes + master_bytes + m_bytes + v_bytes) + activation_bytes
    offloaded = n * (master_bytes + m_bytes + v_bytes)
    return float(offloaded) / float(total_before)


def grade(sol, fx) -> dict:
    """
    Random model sizes / dtype byte-widths / activation memory; compares the
    submitted fractional VRAM reduction to the closed-form offloaded/total
    ratio.
    """
    rng = np.random.default_rng(0)
    worst = 0.0

    for _ in range(8):
        n_params = int(rng.integers(1_000, 5_000_000_000))
        param_bytes = int(rng.choice([1, 2, 4]))
        grad_bytes = int(rng.choice([1, 2, 4]))
        master_bytes = int(rng.choice([2, 4]))
        m_bytes = int(rng.choice([2, 4]))
        v_bytes = int(rng.choice([2, 4]))
        activation_bytes = int(rng.integers(0, 10_000_000_000))

        expected = _oracle(n_params, param_bytes, grad_bytes, master_bytes, m_bytes, v_bytes, activation_bytes)
        try:
            got = sol.vram_reduction_from_offload(
                n_params, param_bytes, grad_bytes, master_bytes, m_bytes, v_bytes, activation_bytes
            )
            err = scorers.rel_err(np.array([expected]), np.array([float(got)]))
        except Exception:
            return {"rel_err": float("inf")}

        worst = max(worst, err)

    return {"rel_err": worst}
