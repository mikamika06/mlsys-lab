import ref

def check(workdir):
    from isa.throughput import analyze_throughput
    out = {"configs_matched": 0.0, "configs": 0.0}

    configs = [
        ("scalar", 64, 2),
        ("avx2", 256, 2),
        ("avx512_vnni", 512, 2),
        ("avx2", 256, 1)
    ]
    out["configs"] = float(len(configs))
    ok = 0

    for cfg in configs:
        want = ref.analyze_throughput(*cfg)
        got = analyze_throughput(*cfg)
        if want == got:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"failed for {cfg}: want {want}, got {got}"

    out["configs_matched"] = float(ok)
    return out
