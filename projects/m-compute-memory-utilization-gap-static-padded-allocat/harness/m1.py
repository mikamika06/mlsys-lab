import ref


def check(workdir):
    from gap.utilization import compute_utilization_gap

    test_cases = [
        ([100, 200, 300], 1024, 16, 512),
        ([50, 50, 50, 50], 512, 32, 128),
        ([1, 1000, 500, 250, 10], 2048, 16, 2048),
        ([], 1024, 16, 512),
    ]

    out = {"utilization_metrics_matched": 0.0}
    matched = 0

    for seq_lengths, max_len, block_size, bpt in test_cases:
        want = ref.ref_compute_utilization_gap(seq_lengths, max_len, block_size, bpt)
        try:
            got = compute_utilization_gap(seq_lengths, max_len, block_size, bpt)
        except Exception as e:
            out["_note"] = f"Error evaluating compute_utilization_gap: {str(e)}"
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
        out["utilization_metrics_matched"] = 1.0

    return out
