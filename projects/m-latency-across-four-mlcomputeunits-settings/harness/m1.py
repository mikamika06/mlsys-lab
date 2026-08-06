import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)

    try:
        from ane_diag.profile import evaluate_all_units
    except ImportError:
        return {
            "latency_ratio": 0.0,
            "_note": "Could not import ane_diag.profile.evaluate_all_units",
        }

    graph = ref.generate_test_graph()
    want = ref.evaluate_all_units(graph)

    try:
        got = evaluate_all_units(graph)
    except Exception as e:
        return {
            "latency_ratio": 0.0,
            "_note": f"evaluate_all_units raised exception: {e}",
        }

    if not isinstance(got, dict):
        return {
            "latency_ratio": 0.0,
            "_note": f"Expected dict return, got {type(got)}",
        }

    matches = 0
    total = len(want)
    for k, v in want.items():
        if k in got and abs(got[k] - v) < 1e-5:
            matches += 1

    ratio = float(matches) / float(total) if total > 0 else 0.0
    out = {"latency_ratio": ratio}
    if ratio < 1.0:
        out["_note"] = f"Matched {matches}/{total} settings. Got: {got}, Want: {want}"
    return out
