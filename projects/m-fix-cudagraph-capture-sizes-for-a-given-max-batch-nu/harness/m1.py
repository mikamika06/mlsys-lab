import ref


def check(workdir):
    from speculative.cudagraph import compute_capture_sizes

    out = {"sizes_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        b = cfg["max_batch_size"]
        n = cfg["num_speculative_tokens"]
        want = ref.compute_capture_sizes(b, n)
        got = compute_capture_sizes(b, n)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["sizes_matched"] = float(ok)
    return out
