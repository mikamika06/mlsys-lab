import ref


def check(workdir):
    from swm.memory import kv_cache_memory_bytes

    out = {"mem_correct": 0.0}
    configs = ref.get_m2_configs()
    ok = 0
    for i, cfg in enumerate(configs):
        want = ref.kv_cache_memory_bytes(*cfg)
        got = kv_cache_memory_bytes(*cfg)
        if want == got:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"

    if ok == len(configs):
        out["mem_correct"] = 1.0

    return out
