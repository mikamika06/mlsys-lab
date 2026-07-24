def zero_stage_bytes(num_params: int, num_devices: int, stage: int) -> float:
    """Per-device memory footprint in bytes under ZeRO stage `stage` (1, 2,
    or 3), for mixed-precision Adam training with `num_params` parameters
    replicated/sharded across `num_devices` data-parallel devices.

    Assumes fp16 parameters (2 bytes/param), fp16 gradients (2 bytes/param),
    and fp32 Adam optimizer state consisting of a master fp32 parameter
    copy, momentum, and variance (4 + 4 + 4 = 12 bytes/param) -- the
    standard ZeRO paper accounting (16 bytes/param total, unsharded).

    - Stage 1 (P_os): only the optimizer states (12 bytes/param) are
      partitioned across `num_devices`.
    - Stage 2 (P_os+g): optimizer states AND gradients (2 bytes/param) are
      partitioned.
    - Stage 3 (P_os+g+p): everything, including the parameters themselves
      (2 bytes/param), is partitioned.
    """
    P = float(num_params)
    N = float(num_devices)

    if stage == 1:
        params_term = 2.0 * P
        grads_term = 2.0 * P
        opt_term = 12.0 * P / N
    elif stage == 2:
        params_term = 2.0 * P
        grads_term = 2.0 * P / N
        opt_term = 12.0 * P / N
    elif stage == 3:
        params_term = 2.0 * P / N
        grads_term = 2.0 * P / N
        opt_term = 12.0 * P / N
    else:
        raise ValueError(f"unknown stage {stage!r}")

    return params_term + grads_term + opt_term
