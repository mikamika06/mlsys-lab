import numpy as np


def _ref_bytes(num_params, num_devices, stage):
    """Real oracle: the standard ZeRO per-device memory formula for
    mixed-precision Adam training (Rajbhandari et al., 2019).

    Baseline per-replica cost (Psi = num_params):
      fp16 params: 2*Psi bytes, fp16 grads: 2*Psi bytes,
      fp32 optimizer states (master copy + momentum + variance): 12*Psi bytes.
      Total unsharded: 16*Psi bytes.

    Stage 1 (P_os): only the 12*Psi optimizer-state term is partitioned
      across `num_devices` (N) devices.
        per_device = 4*Psi + 12*Psi/N
    Stage 2 (P_os+g): optimizer states AND the 2*Psi gradient term are
      partitioned.
        per_device = 2*Psi + 14*Psi/N
    Stage 3 (P_os+g+p): everything (params, grads, optimizer states) is
      partitioned.
        per_device = 16*Psi/N
    """
    P = float(num_params)
    N = float(num_devices)
    if stage == 1:
        return 4.0 * P + 12.0 * P / N
    if stage == 2:
        return 2.0 * P + 14.0 * P / N
    if stage == 3:
        return 16.0 * P / N
    raise ValueError(f"unknown stage {stage!r}")


def _cases():
    cases = []
    for P in [1, 100, 1_000, 1_000_000, 7_000_000_000]:
        for N in [1, 2, 4, 8, 16, 64]:
            for stage in (1, 2, 3):
                cases.append((P, N, stage))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for P, N, stage in _cases():
        expected = _ref_bytes(P, N, stage)
        try:
            got = float(sol.zero_stage_bytes(P, N, stage))
        except Exception:
            return {"size_ratio": float("inf")}

        if not np.isfinite(got):
            return {"size_ratio": float("inf")}

        dev = abs(got / expected - 1.0)
        worst = max(worst, dev)

    return {"size_ratio": worst}
