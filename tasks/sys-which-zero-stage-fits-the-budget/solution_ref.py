def _stage_bytes(psi, n_devices, stage):
    if stage == 0:
        return 16.0 * psi
    if stage == 1:
        return 4.0 * psi + 12.0 * psi / n_devices
    if stage == 2:
        return 2.0 * psi + 14.0 * psi / n_devices
    if stage == 3:
        return 16.0 * psi / n_devices
    raise ValueError(stage)


def min_zero_stage(psi: float, n_devices: int, budget_bytes: float) -> int:
    """
    Smallest ZeRO stage in {0,1,2,3} whose per-device memory requirement
    (mixed-precision Adam: fp16 params+grads, fp32 master+momentum+variance)
    fits within budget_bytes; -1 if none of the four stages fit.
    """
    for stage in (0, 1, 2, 3):
        if _stage_bytes(psi, n_devices, stage) <= budget_bytes:
            return stage
    return -1
