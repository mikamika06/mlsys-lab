import ref

def check(workdir):
    from mlxflow.bench import compute_speedup

    runs_fp16 = [10.5, 11.2, 10.8]
    runs_quant = [18.1, 18.5, 18.3]
    want = ref.compute_speedup(runs_fp16, runs_quant)
    try:
        got = compute_speedup(runs_fp16, runs_quant)
    except Exception:
        got = None

    out = {"ratio_matched": 1.0 if got == want else 0.0}
    if got != want:
        out["_note"] = f"got {got}, want {want}"
    return out
