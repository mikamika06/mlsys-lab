import numpy as np

def _oracle(matrix, rank):
    """SVD oracle: compute reference reconstruction error and energy fraction."""
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    s_sq = s ** 2
    total_energy = float(np.sum(s_sq))
    r = min(max(rank, 0), len(s))
    recon_err = float(np.sum(s_sq[r:]))
    energy_frac = float(np.sum(s_sq[:r]) / total_energy) if total_energy > 0 else 0.0
    return recon_err, energy_frac

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(42)
    test_cases = [
        (rng.randn(20, 10), 3),       # tall matrix, moderate rank
        (rng.randn(50, 50), 10),      # square matrix
        (rng.randn(100, 20), 5),      # very tall, small rank
        (np.ones((15, 8)), 4),        # rank-1 matrix, rank=4 > actual rank
        (rng.randn(30, 30), 30),      # full-rank edge case
        (rng.randn(10, 40), 0),       # rank-0: everything discarded
        (rng.randn(8, 8), 8),         # full-rank square, rank == k
    ]

    worst = 0.0
    for matrix, rank in test_cases:
        try:
            got_err, got_energy = sol.kv_reconstruction_error_and_energy(
                matrix.copy(), rank
            )
            ref_err, ref_energy = _oracle(matrix, rank)
        except Exception:
            return {"rel_err": 1.0}

        for got, ref in [(got_err, ref_err), (got_energy, ref_energy)]:
            if abs(ref) > 1e-12:
                worst = max(worst, abs(got - ref) / abs(ref))
            else:
                worst = max(worst, abs(got - ref))

    return {"rel_err": worst}
