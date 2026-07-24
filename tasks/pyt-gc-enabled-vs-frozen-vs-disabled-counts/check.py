import gc


def _workload():
    items = []
    for _ in range(3000):
        a = []
        a.append(a)
        items.append(a)
    return items


def _count_delta(before):
    after = gc.get_stats()
    return sum(
        after[i]["collections"] - before[i]["collections"]
        for i in range(len(after))
    )


def _run_mode(mode):
    old_enabled = gc.isenabled()
    old_threshold = gc.get_threshold()
    try:
        gc.collect()
        gc.set_threshold(20, 5, 5)

        if mode == "enabled":
            gc.enable()
        elif mode == "frozen":
            gc.enable()
            gc.freeze()
        elif mode == "disabled":
            gc.disable()

        before = gc.get_stats()
        _workload()
        return _count_delta(before)
    finally:
        if hasattr(gc, "unfreeze"):
            gc.unfreeze()
        gc.set_threshold(*old_threshold)
        if old_enabled:
            gc.enable()
        else:
            gc.disable()


def _oracle():
    return (
        _run_mode("enabled"),
        _run_mode("frozen"),
        _run_mode("disabled"),
    )


def grade(sol, fx) -> dict:
    try:
        ref = _oracle()
        got = tuple(sol.gc_collection_counts())
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == ref else 0.0}
