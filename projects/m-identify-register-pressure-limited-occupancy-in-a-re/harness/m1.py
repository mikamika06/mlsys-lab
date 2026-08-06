import ref


def check(workdir):
    from profiler.occupancy import compute_occupancy
    out = {"occupancy_matched": 0.0}
    ok = True
    for i, p in enumerate(ref.NCU_PROFILES):
        want = ref.compute_occupancy(p["regs_per_thread"], p["shmem_per_block"], p["block_size"], p["device_specs"])
        got = compute_occupancy(p["regs_per_thread"], p["shmem_per_block"], p["block_size"], p["device_specs"])
        if got is None or abs(got.get("occupancy", -1) - want["occupancy"]) > 1e-4 or got.get("limiting_factor") != want["limiting_factor"]:
            ok = False
            out["_note"] = f"profile {i}: got {got}, want {want}"
            break
    if ok:
        out["occupancy_matched"] = 1.0
    return out
