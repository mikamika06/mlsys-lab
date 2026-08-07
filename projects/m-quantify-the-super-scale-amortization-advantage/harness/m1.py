import ref


def check(workdir):
    from kquant.amortization import compute_superblock_footprint

    out = {"footprint_rel_err": 1.0}
    max_err = 0.0

    for cfg in ref.CONFIGS:
        want = ref.compute_superblock_footprint(**cfg)
        got = compute_superblock_footprint(**cfg)

        for k in ["num_superblocks", "total_bytes", "metadata_ratio", "bits_per_weight"]:
            if k not in got:
                out["_note"] = f"missing key {k} in footprint output"
                return out
            ref_val = float(want[k])
            got_val = float(got[k])
            err = abs(got_val - ref_val) / max(abs(ref_val), 1e-9)
            if err > max_err:
                max_err = err

    out["footprint_rel_err"] = float(max_err)
    return out
