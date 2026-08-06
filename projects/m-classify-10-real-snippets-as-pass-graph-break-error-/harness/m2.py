import ref


def check(workdir):
    from compiletracer.dynamic_shapes import track_shape_compilations
    from compiletracer.backend_overhead import isolate_backend_overhead

    out = {
        "static_overhead_match": 0.0,
        "dynamic_overhead_match": 0.0,
        "backend_breakdown_match": 0.0
    }

    want_static = ref.track_shape_compilations(ref.SHAPE_SEQUENCE, dynamic=False)
    want_dynamic = ref.track_shape_compilations(ref.SHAPE_SEQUENCE, dynamic=True)
    want_overhead = ref.isolate_backend_overhead(ref.TRACE_TIMINGS)

    try:
        got_static = track_shape_compilations(ref.SHAPE_SEQUENCE, dynamic=False)
        if got_static == want_static:
            out["static_overhead_match"] = 1.0
    except Exception as e:
        out["_note"] = f"track_shape_compilations static failed: {e}"

    try:
        got_dynamic = track_shape_compilations(ref.SHAPE_SEQUENCE, dynamic=True)
        if got_dynamic == want_dynamic:
            out["dynamic_overhead_match"] = 1.0
    except Exception as e:
        out["_note"] = f"track_shape_compilations dynamic failed: {e}"

    try:
        got_overhead = isolate_backend_overhead(ref.TRACE_TIMINGS)
        if got_overhead == want_overhead:
            out["backend_breakdown_match"] = 1.0
    except Exception as e:
        out["_note"] = f"isolate_backend_overhead failed: {e}"

    return out
