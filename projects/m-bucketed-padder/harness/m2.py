import ref


def check(workdir):
    out = {"waste_computation_match": 0.0, "optimal_ladder_match": 0.0}

    try:
        from padder.cost import compute_padding_waste
        from padder.ladder import find_optimal_ladder
    except ImportError as e:
        out["_note"] = f"Import error: {e}"
        return out

    lengths = ref.generate_lengths(seed=99, n=200, max_len=180)
    boundaries = [32, 64, 128, 256]

    want_waste, want_ratio = ref.ref_compute_padding_waste(lengths, boundaries)
    try:
        got_waste, got_ratio = compute_padding_waste(lengths, boundaries)
        if got_waste == want_waste and abs(got_ratio - want_ratio) < 1e-6:
            out["waste_computation_match"] = 1.0
        else:
            out["_note"] = f"Waste mismatch: got ({got_waste}, {got_ratio:.4f}), want ({want_waste}, {want_ratio:.4f})"
    except Exception as e:  # noqa: BLE001
        out["_note"] = f"compute_padding_waste error: {e}"
        return out

    candidate_bounds = [16, 32, 48, 64, 96, 128, 160, 192, 224, 256]
    max_buckets = 3
    compilation_cost = 50.0
    alignment = 16

    want_ladder, want_cost = ref.ref_find_optimal_ladder(
        lengths, candidate_bounds, max_buckets, compilation_cost, alignment
    )

    try:
        got_ladder, got_cost = find_optimal_ladder(
            lengths, candidate_bounds, max_buckets, compilation_cost, alignment
        )
        if got_ladder == want_ladder and abs(got_cost - want_cost) < 1e-5:
            out["optimal_ladder_match"] = 1.0
        else:
            out["_note"] = f"Ladder mismatch: got ({got_ladder}, {got_cost}), want ({want_ladder}, {want_cost})"
    except Exception as e:  # noqa: BLE001
        out["_note"] = f"find_optimal_ladder error: {e}"

    return out
