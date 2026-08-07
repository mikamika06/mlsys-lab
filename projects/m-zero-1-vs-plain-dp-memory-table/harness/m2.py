import ref


def check(workdir):
    from zerodp.partition import partition_flat_contiguous

    out = {"flat_partitions_matched": 0.0, "total": float(len(ref.CONFIGS_M2))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS_M2):
        want = ref.partition_flat_contiguous(**cfg)
        got = partition_flat_contiguous(**cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"cfg {i}: got {got}, want {want}"
    out["flat_partitions_matched"] = (
        1.0 if ok == len(ref.CONFIGS_M2) else 0.0
    )
    return out
