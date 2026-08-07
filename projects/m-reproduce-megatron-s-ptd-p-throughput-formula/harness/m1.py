import ref


def check(workdir):
    from pipelp.throughput import compute_throughput

    out = {"throughput_matched": 0.0}
    ok = True
    for cfg in ref.CONFIGS:
        want = ref.compute_throughput(
            cfg["pp_size"], cfg["microbatches"], cfg["total_flops"], cfg["time_per_stage"]
        )
        got = compute_throughput(
            cfg["pp_size"], cfg["microbatches"], cfg["total_flops"], cfg["time_per_stage"]
        )
        if abs(got - want) > 1e-5:
            ok = False
            out["_note"] = f"mismatch for cfg {cfg}: got {got}, want {want}"
            break
    if ok:
        out["throughput_matched"] = 1.0
    return out
