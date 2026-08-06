import ref

def check(workdir):
    from quantlib.crossover import measure_scheme
    out = {"metrics_matched": 0.0}
    ok = 0
    for w in ref.WORKLOADS:
        for s in ref.SCHEMES:
            want = ref.measure_scheme(s, w)
            try:
                got = measure_scheme(s, w)
            except Exception:
                continue
            if isinstance(got, dict) and abs(got.get("throughput", 0) - want["throughput"]) < 1e-5:
                ok += 1
    out["metrics_matched"] = float(ok)
    return out
