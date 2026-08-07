import ref

def check(workdir):
    from tpscaling.benchmark import run_benchmark
    out = {"benchmarks_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        for tp in [1, 2]:
            want = ref.run_benchmark(tp, cfg)
            try:
                got = run_benchmark(tp, cfg)
            except Exception as e:
                if "_note" not in out:
                    out["_note"] = f"config {i} tp {tp} raised {type(e).__name__}: {str(e)[:100]}"
                continue
            if isinstance(got, dict) and abs(got.get("throughput", 0) - want["throughput"]) < 1e-5:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"config {i} tp {tp}: got {got}, reference {want}"
    out["benchmarks_matched"] = float(ok)
    return out
