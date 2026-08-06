import ref


def check(workdir):
    from profiler.tiling import verify_tiling_dram_reduction

    cases = ref.generate_tiling_cases()
    dram_ok = 0
    speedup_ok = 0
    total = len(cases)

    for c in cases:
        res = verify_tiling_dram_reduction(
            c["naive_dram_bytes"],
            c["tiled_dram_bytes"],
            c["measured_speedup"]
        )
        if res.get("dram_bytes_reduced") is True:
            dram_ok += 1
        if res.get("speedup_explained_by_dram") is not None:
            speedup_ok += 1

    out = {
        "dram_bytes_verified": 1.0 if dram_ok == total else 0.0,
        "speedup_explained": 1.0 if speedup_ok == total else 0.0
    }
    return out
