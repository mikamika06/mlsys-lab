WARP = 32


def _reference(regs_per_thread, shared_bytes_per_block, threads_per_block,
                max_regs_per_sm, max_shared_bytes_per_sm, max_threads_per_sm, max_blocks_per_sm):
    """Closed-form occupancy calculation: the number of concurrently
    resident blocks on one SM is the minimum of what each resource limit
    (registers, shared memory, thread-count) allows on its own, capped by
    the hardware's flat max-blocks-per-SM limit.
    """
    blocks_by_regs = max_regs_per_sm // (regs_per_thread * threads_per_block)
    if shared_bytes_per_block > 0:
        blocks_by_shared = max_shared_bytes_per_sm // shared_bytes_per_block
    else:
        blocks_by_shared = max_blocks_per_sm
    blocks_by_threads = max_threads_per_sm // threads_per_block

    active_blocks = min(blocks_by_regs, blocks_by_shared, blocks_by_threads, max_blocks_per_sm)

    warps_per_block = (threads_per_block + WARP - 1) // WARP
    active_warps = active_blocks * warps_per_block

    max_warps_per_sm = max_threads_per_sm // WARP
    occupancy = active_warps / max_warps_per_sm

    return active_warps, occupancy


def grade(sol, fx) -> dict:
    """Grade compute_occupancy against the closed-form reference across
    configurations chosen so each resource (registers, shared memory,
    thread count, and the flat block-count cap) is the binding limiter
    in at least one case.
    """
    sm_limits = dict(
        max_regs_per_sm=65536,
        max_shared_bytes_per_sm=49152,
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
    )
    # (regs_per_thread, shared_bytes_per_block, threads_per_block)
    test_cases = [
        (64, 1024, 256),    # register-bound
        (16, 16384, 128),   # shared-memory-bound
        (8, 2048, 1024),    # thread-count-bound
        (2, 64, 32),        # flat block-count-cap bound
        (32, 4096, 256),    # exactly 100% occupancy
    ]

    max_diff = 0.0
    for regs, shared, tpb in test_cases:
        args = (regs, shared, tpb, *sm_limits.values())
        try:
            learner_val = sol.compute_occupancy(*args)
        except Exception:
            return {"exact_match": 0.0}

        ref_val = _reference(*args)
        if tuple(learner_val) != ref_val:
            max_diff = 1.0

    return {"exact_match": 0.0 if max_diff else 1.0}
