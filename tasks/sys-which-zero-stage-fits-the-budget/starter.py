def min_zero_stage(psi: float, n_devices: int, budget_bytes: float) -> int:
    """
    Given a model with `psi` parameters trained with mixed-precision Adam
    across `n_devices` data-parallel devices, find the smallest ZeRO
    stage in {0, 1, 2, 3} whose per-device memory requirement fits
    within `budget_bytes`. Return -1 if none of the four stages fit.

    Per-device memory (bytes), with Psi = psi, N = n_devices:
      stage 0 (no ZeRO):               16*Psi
      stage 1 (optimizer state shard): 4*Psi + 12*Psi/N
      stage 2 (+ gradient shard):      2*Psi + 14*Psi/N
      stage 3 (+ parameter shard):     16*Psi/N
    """
    raise NotImplementedError('your code here')
