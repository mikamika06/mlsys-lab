import ref


def check(workdir):
    from goodput.metrics import compute_e2el_gap
    traces = ref.generate_traces()
    want = ref.compute_e2el_gap(traces)
    got = compute_e2el_gap(traces)
    out = {"gap_matched": 0.0}
    if abs(got - want) < 1e-5:
        out["gap_matched"] = 1.0
    else:
        out["_note"] = f"got gap {got}, want {want}"
    return out
