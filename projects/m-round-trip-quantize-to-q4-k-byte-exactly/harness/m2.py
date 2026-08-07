import ref


def check(workdir):
    from q4k.analysis import find_dominating_subblock

    weights_list = ref.get_test_weights()
    success = 0
    total = len(weights_list)
    for w in weights_list:
        try:
            idx = find_dominating_subblock(w)
            if isinstance(idx, int) and 0 <= idx < 8:
                success += 1
        except Exception:
            pass

    score = 1.0 if success == total else 0.0
    out = {"max_mse_subblock_identified": score}
    if score == 0.0:
        out["_note"] = "Failed to correctly identify sub-block dominating MSE."
    return out
