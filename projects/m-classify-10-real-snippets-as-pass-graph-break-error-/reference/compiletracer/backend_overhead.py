def isolate_backend_overhead(timings):
    """Isolate compile-time-only overhead of Inductor vs eager tracing."""
    raw_eager = timings["raw_eager"]
    aot_eager = timings["aot_eager"]
    inductor = timings["inductor"]

    raw_mean = sum(raw_eager) / float(len(raw_eager))

    aot_first = float(aot_eager[0])
    aot_warm_mean = sum(aot_eager[1:]) / float(len(aot_eager[1:]))
    tracing_overhead = max(0.0, aot_first - aot_warm_mean)

    ind_first = float(inductor[0])
    ind_warm_mean = sum(inductor[1:]) / float(len(inductor[1:]))
    total_ind_overhead = max(0.0, ind_first - ind_warm_mean)

    ind_compile_only = max(0.0, total_ind_overhead - tracing_overhead)
    speedup = raw_mean / ind_warm_mean if ind_warm_mean > 0 else 1.0

    return {
        "tracing_overhead": round(tracing_overhead, 6),
        "total_inductor_overhead": round(total_ind_overhead, 6),
        "inductor_compile_only_overhead": round(ind_compile_only, 6),
        "speedup_ratio": round(speedup, 6)
    }
