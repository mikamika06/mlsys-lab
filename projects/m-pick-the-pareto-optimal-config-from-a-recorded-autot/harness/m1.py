import ref


def check(workdir):
    from autotune.sweep import load_sweep

    raw = ref.raw_sweep_text()
    want = ref.load_sweep(raw)
    got = load_sweep(raw)

    out = {"records_matched": 0.0, "records": float(len(want))}
    if got == want:
        out["records_matched"] = float(len(want))
    else:
        out["_note"] = f"got {got[:1]}, want {want[:1]}"
    return out
