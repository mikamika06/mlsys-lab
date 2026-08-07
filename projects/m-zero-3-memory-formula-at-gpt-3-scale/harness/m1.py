import ref

def check(workdir):
    from zerothree.memory import calculate_zero3_memory

    out = {"memory_matches": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.calculate_zero3_memory(cfg["num_params"], cfg["bytes_per_param"], cfg["dp_degree"])
        try:
            got = calculate_zero3_memory(cfg["num_params"], cfg["bytes_per_param"], cfg["dp_degree"])
        except Exception as e:
            out["_note"] = f"config {i} raised {type(e).__name__}: {str(e)[:100]}"
            return out

        if isinstance(got, (int, float)) and abs(got - want) < 1e-2:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"

    out["memory_matches"] = float(ok)
    return out
