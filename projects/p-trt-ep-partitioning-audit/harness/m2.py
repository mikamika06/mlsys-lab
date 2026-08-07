import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref
    import trtep.audit as audit

    m = {"breakers_api_ok": 0.0, "breakers_matched": 0.0}
    try:
        g = ref.build_benchmark_graph()
        breakers = audit.find_partition_breakers(g, ref.DEFAULT_SUPPORTED_OPS)
    except Exception:
        return m

    if not isinstance(breakers, list):
        return m

    m["breakers_api_ok"] = 1.0

    expected = ref.ref_find_breakers(g, ref.DEFAULT_SUPPORTED_OPS)
    if sorted(breakers) == sorted(expected):
        m["breakers_matched"] = 1.0

    return m
