import ref

def check(workdir):
    from bench.harness import BenchmarkHarness
    out = {"harness_matched": 0.0}
    try:
        h = BenchmarkHarness(prompt_len=128, seq_len=256, dtype="float16", backend="mlx")
        lats = h.run(warmup=5, iters=20)
        if isinstance(lats, list) and len(lats) == 20:
            out["harness_matched"] = 3.0
    except Exception as e:
        out["_note"] = str(e)
    return out
