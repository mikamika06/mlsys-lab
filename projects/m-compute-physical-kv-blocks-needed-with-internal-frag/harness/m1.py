import ref


def check(workdir):
    from paged_kv.allocator import compute_physical_blocks_needed

    out = {"blocks_matched": 0.0, "frag_ratio_matched": 0.0}

    test_cases = [
        ([1, 15, 16, 17, 32, 33], 16),
        ([100, 250, 500, 1024], 32),
        ([0, 5, 128], 16),
        ([2048, 4096, 8192], 64),
    ]

    blocks_ok = 0
    frag_ok = 0
    total = len(test_cases)

    for i, (seq_lens, block_size) in enumerate(test_cases):
        want = ref.compute_physical_blocks_needed(seq_lens, block_size)
        try:
            got = compute_physical_blocks_needed(seq_lens, block_size)
        except Exception as e:
            out["_note"] = f"case {i} raised {type(e).__name__}: {e}"
            return out

        if got.get("total_blocks") == want["total_blocks"] and got.get("total_capacity") == want["total_capacity"]:
            blocks_ok += 1
        if abs(got.get("fragmentation_ratio", -1.0) - want["fragmentation_ratio"]) < 1e-5:
            frag_ok += 1

    if blocks_ok == total:
        out["blocks_matched"] = 1.0
    if frag_ok == total:
        out["frag_ratio_matched"] = 1.0

    return out
