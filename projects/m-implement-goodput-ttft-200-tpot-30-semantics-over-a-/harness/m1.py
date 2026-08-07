import ref


def check(workdir):
    from goodput.metrics import compute_goodput
    traces = ref.generate_traces()
    want = ref.compute_goodput(traces, 200.0, 30.0)
    got = compute_goodput(traces, 200.0, 30.0)
    out = {"goodput_matched": 0.0}
    if abs(got - want) < 1e-5:
        out["goodput_matched"] = 1.0
    else:
        out["_note"] = f"got goodput {got}, want {want}"
    return out
