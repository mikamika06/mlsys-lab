def _reference(block_dim, regs_per_thread, shared_bytes_per_block):
    max_regs_per_sm = 65536
    max_shared_bytes = 49152
    max_threads_per_sm = 2048

    threads_by_regs = min(max_regs_per_sm // regs_per_thread,
                          max_threads_per_sm)

    if shared_bytes_per_block <= 0:
        blocks_by_shared = max_threads_per_sm  # effectively unlimited
    else:
        blocks_by_shared = max(1, max_shared_bytes // shared_bytes_per_block)
    threads_by_shared = min(blocks_by_shared * block_dim,
                            max_threads_per_sm)

    threads_by_thread_cap = max_threads_per_sm

    limits = {
        'register': threads_by_regs,
        'shared':   threads_by_shared,
        'thread':   threads_by_thread_cap
    }
    # Find the resource with minimal allowed threads
    limiting_key = min(limits, key=limits.get)
    return limiting_key


def grade(sol, fx) -> dict:
    ok = 1.0
    try:
        result = sol.identify_limiter(128, 32, 0)
        if result != _reference(128, 32, 0):
            ok = 0.0
    except Exception:
        ok = 0.0

    return {"exact_match": ok}
