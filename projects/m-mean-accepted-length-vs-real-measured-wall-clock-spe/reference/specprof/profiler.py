import time


def record_speculative_step(
    draft_fn, target_fn, verify_fn, draft_args=(), target_args=(), verify_args=()
):
    """Executes draft, target, and verify phases while recording timestamps."""
    t0 = time.perf_counter()
    draft_res = draft_fn(*draft_args)
    t1 = time.perf_counter()

    target_res = target_fn(*target_args)
    t2 = time.perf_counter()

    verify_res = verify_fn(*verify_args)
    t3 = time.perf_counter()

    events = [
        {
            "name": "draft",
            "cat": "draft",
            "dur": (t1 - t0) * 1e6,
            "ts": t0 * 1e6,
        },
        {
            "name": "target",
            "cat": "target",
            "dur": (t2 - t1) * 1e6,
            "ts": t1 * 1e6,
        },
        {
            "name": "verify",
            "cat": "verify",
            "dur": (t3 - t2) * 1e6,
            "ts": t2 * 1e6,
        },
    ]

    return (draft_res, target_res, verify_res), events
