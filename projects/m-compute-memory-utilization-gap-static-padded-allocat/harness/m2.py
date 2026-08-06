import ref


def check(workdir):
    from gap.probing import probe_max_concurrency

    test_cases = [
        (1024 * 1024 * 100, [100] * 500, 16, 512, 0.05),
        (1024 * 1024 * 50, [256, 512, 128, 64] * 100, 32, 1024, 0.10),
        (1024 * 100, [10] * 10, 16, 256, 0.0),
    ]

    out = {"probing_results_matched": 0.0}
    matched = 0

    for total_mem, seq_lens, block_size, bpt, margin in test_cases:
        want = ref.ref_probe_max_concurrency(total_mem, iter(seq_lens), block_size, bpt, margin)
        try:
            got = probe_max_concurrency(total_mem, iter(seq_lens), block_size, bpt, margin)
        except Exception as e:
            out["_note"] = f"Error evaluating probe_max_concurrency: {str(e)}"
            return out

        is_match = True
        for k in want:
            if abs(want[k] - got.get(k, 0)) > 1e-4:
                is_match = False
                out["_note"] = f"Mismatch in {k}: got {got.get(k)}, expected {want[k]}"
                break

        if is_match:
            matched += 1

    if matched == len(test_cases):
        out["probing_results_matched"] = 1.0

    return out
