import ref


def check(workdir):
    from jaxpr_utils.analyzer import list_unique_primitives, safe_trace_collector

    prims_ok = 0
    for jaxpr in ref.SAMPLE_JAXPRS:
        want = ref.list_unique_primitives(jaxpr)
        got = list_unique_primitives(jaxpr)
        if want == got:
            prims_ok += 1

    leak_ok = True
    try:
        def dummy(x):
            return x * 2
        res = safe_trace_collector(dummy, [1, 2, 3])
        if res != [2, 4, 6]:
            leak_ok = False
    except Exception:
        leak_ok = False

    out = {
        "primitives_matched": 1.0 if prims_ok == len(ref.SAMPLE_JAXPRS) else 0.0,
        "no_leaked_tracer": 1.0 if leak_ok else 0.0
    }
    return out
