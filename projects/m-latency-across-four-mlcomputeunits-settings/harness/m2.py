import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)

    try:
        from ane_diag.fallback import find_cpu_fallback_op
    except ImportError:
        return {
            "isolated_correctly": 0.0,
            "_note": "Could not import ane_diag.fallback.find_cpu_fallback_op",
        }

    graph = ref.generate_test_graph()
    want = ref.find_cpu_fallback_op(graph)

    try:
        got = find_cpu_fallback_op(graph)
    except Exception as e:
        return {
            "isolated_correctly": 0.0,
            "_note": f"find_cpu_fallback_op raised exception: {e}",
        }

    if got == want:
        return {"isolated_correctly": 1.0}
    return {
        "isolated_correctly": 0.0,
        "_note": f"Isolated op '{got}', expected '{want}'",
    }
