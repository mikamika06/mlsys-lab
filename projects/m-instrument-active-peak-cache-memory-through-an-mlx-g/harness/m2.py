import ref

def check(workdir):
    from mlx_mem.throughput import measure_throughput
    out = {"penalty_match": 0.0}
    try:
        t_def = measure_throughput(50, False)
        t_lim = measure_throughput(50, True)
        ref_def = ref.ref_throughput(50, False)
        ref_lim = ref.ref_throughput(50, True)
        if abs(t_def - ref_def) < 1e-5 and abs(t_lim - ref_lim) < 1e-5 and t_lim < t_def:
            out["penalty_match"] = 1.0
        else:
            out["_note"] = f"throughput values or penalty relationship incorrect"
    except Exception as e:
        out["_note"] = str(e)
    return out
