import importlib.util
import math
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_first_fit_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import arena.planner as p
    good_planner = p.plan_activation_arena

    def broken_planner(buffers, default_alignment=64):
        if not buffers:
            return {"arena_size": 0, "offsets": {}}
        placed = []
        for buf in buffers:
            buf_id = buf["id"]
            size = buf["size"]
            start, end = buf["liveness"]
            align = buf.get("alignment", default_alignment)
            cand_offset = 0
            while True:
                conflict = False
                for prev in placed:
                    p_start, p_end = prev["liveness"]
                    if max(start, p_start) <= min(end, p_end):
                        p_off = prev["offset"]
                        p_size = prev["size"]
                        if not (cand_offset + size <= p_off or p_off + p_size <= cand_offset):
                            conflict = True
                            break
                if not conflict:
                    break
                cand_offset += align
            placed.append({"id": buf_id, "offset": cand_offset, "size": size, "liveness": (start, end)})
        max_end = max(item["offset"] + item["size"] for item in placed) if placed else 0
        arena_size = math.ceil(max_end / default_alignment) * default_alignment if max_end > 0 else 0
        return {"arena_size": arena_size, "offsets": {item["id"]: item["offset"] for item in placed}}

    p.plan_activation_arena = broken_planner
    import arena
    arena.plan_activation_arena = broken_planner
    try:
        out["catches_first_fit_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        p.plan_activation_arena = good_planner
        arena.plan_activation_arena = good_planner

    return out
