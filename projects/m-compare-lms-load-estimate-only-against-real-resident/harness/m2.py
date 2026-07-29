import ref


def check(workdir):
    from lmsmem import estimate_bytes, resident_bytes, relative_error, offload_split

    out = {"resident_matched": 0.0, "rel_err": 1.0, "offload_invariant": 0.0}
    ok = 0
    max_diff = 0.0
    for i, (cfg, ctx, bpp) in enumerate(ref.CASES):
        want_resident = ref.resident_bytes(cfg, ctx, bpp)
        got_resident = resident_bytes(cfg, ctx, bpp)
        if got_resident == want_resident:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: resident got {got_resident}, reference {want_resident}"

        want_estimate = ref.estimate_bytes(cfg, ctx, bpp)
        got_estimate = estimate_bytes(cfg, ctx, bpp)
        want_err = ref.relative_error(want_estimate, want_resident)
        got_err = relative_error(got_estimate, got_resident)
        diff = abs(got_err - want_err)
        if diff > max_diff:
            max_diff = diff
    out["resident_matched"] = float(ok)
    out["rel_err"] = max_diff

    total_bytes = ref.resident_bytes(ref.CONFIGS[-1], ref.CONTEXTS[-1], ref.BYTES_PER_PARAM[0])
    ratios = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    prev_gpu = -1
    inv_ok = True
    for ratio in ratios:
        split = offload_split(total_bytes, ratio)
        gpu = split.get("gpu_bytes") if isinstance(split, dict) else None
        cpu = split.get("cpu_bytes") if isinstance(split, dict) else None
        if (gpu is None or cpu is None or gpu < 0 or cpu < 0
                or gpu + cpu != total_bytes or gpu < prev_gpu):
            inv_ok = False
            if "_note" not in out:
                out["_note"] = f"offload split broke at ratio {ratio}: {split}"
            break
        prev_gpu = gpu
    out["offload_invariant"] = 1.0 if inv_ok else 0.0
    return out
