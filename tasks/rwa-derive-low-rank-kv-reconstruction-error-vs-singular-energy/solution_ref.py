import numpy as np

def kv_reconstruction_error_and_energy(matrix: np.ndarray, rank: int) -> tuple[float, float]:
    """Compute rank-r reconstruction error and captured singular energy fraction.

    Uses the SVD identity: the squared Frobenius error of the best rank-r
    approximation equals the sum of the discarded squared singular values,
    and the energy fraction is the sum of the kept squared singular values
    divided by the total.
    """
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    s_sq = s ** 2
    total_energy = float(np.sum(s_sq))
    r = min(max(rank, 0), len(s))
    recon_err = float(np.sum(s_sq[r:]))
    energy_frac = float(np.sum(s_sq[:r]) / total_energy) if total_energy > 0 else 0.0
    return recon_err, energy_frac
