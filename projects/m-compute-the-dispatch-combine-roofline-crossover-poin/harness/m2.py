import ref


def check(workdir):
    from moeplan.placement import pack_experts

    out = {"placement_matched": 0.0, "max_load_optimal": 0.0}
    matched = 0
    optimal_count = 0

    for i, cfg in enumerate(ref.PACK_CONFIGS):
        want_pack = ref.ref_pack(cfg)
        try:
            got_pack = pack_experts(cfg["loads"], cfg["ranks"], cfg["exp_mem"], cfg["budget"])
            if got_pack == want_pack:
                matched += 1

            r_loads_got = [0] * cfg["ranks"]
            r_loads_want = [0] * cfg["ranks"]
            for exp_id, r in enumerate(got_pack):
                r_loads_got[r] += cfg["loads"][exp_id]
            for exp_id, r in enumerate(want_pack):
                r_loads_want[r] += cfg["loads"][exp_id]

            if max(r_loads_got) <= max(r_loads_want):
                optimal_count += 1
            elif "_note" not in out:
                out["_note"] = f"cfg {i}: max load got {max(r_loads_got)}, want <= {max(r_loads_want)}"
        except Exception as e:  # noqa: BLE001
            if "_note" not in out:
                out["_note"] = f"cfg {i} raised {type(e).__name__}: {e}"

    out["placement_matched"] = float(matched)
    out["max_load_optimal"] = 1.0 if optimal_count == len(ref.PACK_CONFIGS) else 0.0
    return out
