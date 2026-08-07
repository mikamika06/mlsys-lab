import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from numprec.autocast_rules import predict_autocast_action
    from numprec.overflow import fp16_overflow_tail_probability

    out = {"autocast_matched": 0.0, "overflow_matched": 0.0}

    autocast_ok = True
    for op, want in ref.AUTOCAST_BENCHMARK_OPS.items():
        try:
            got = predict_autocast_action(op)
            if got != want:
                autocast_ok = False
                out["_note"] = f"Op {op}: got {got}, want {want}"
                break
        except Exception as e:
            autocast_ok = False
            out["_note"] = f"Error evaluating op {op}: {e}"
            break

    if autocast_ok:
        out["autocast_matched"] = 1.0

    overflow_ok = True
    for std in ref.STDS_TO_CHECK:
        want_p = ref.ref_fp16_overflow_prob(std)
        try:
            got_p = fp16_overflow_tail_probability(std)
            rel_diff = abs(got_p - want_p) / (want_p + 1e-15)
            if rel_diff > 1e-3 and abs(got_p - want_p) > 1e-6:
                overflow_ok = False
                out["_note"] = f"Overflow prob mismatch for std={std}: got {got_p}, want {want_p}"
                break
        except Exception as e:
            overflow_ok = False
            out["_note"] = f"Error evaluating overflow for std={std}: {e}"
            break

    if overflow_ok:
        out["overflow_matched"] = 1.0

    return out
