def zero_stage_bytes(num_params: int, num_devices: int, stage: int) -> float:
    """Per-device memory footprint in bytes under ZeRO stage `stage` (1, 2,
    or 3), for mixed-precision Adam training with `num_params` parameters
    replicated/sharded across `num_devices` data-parallel devices.

    See task.md for the exact per-stage formula.
    """
    raise NotImplementedError('your code here')
