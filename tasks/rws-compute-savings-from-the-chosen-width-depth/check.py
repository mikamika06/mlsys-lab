def _oracle(layer_shapes, depth_keep, width_keeps):
    teacher = 0
    for inp, out in layer_shapes:
        teacher += inp * out + out

    pruned = 0
    prev_width = None
    for idx, ((inp, out), keep) in enumerate(zip(layer_shapes, depth_keep)):
        if not keep:
            continue
        out_width = len(width_keeps[idx])
        if prev_width is None:
            in_width = inp
        else:
            in_width = prev_width
        pruned += in_width * out_width + out_width
        prev_width = out_width

    ratio = teacher / pruned if pruned else float("inf")
    return pruned, ratio


def grade(sol, fx) -> dict:
    cases = [
        (
            [(8, 16), (16, 32), (32, 4)],
            [True, False, True],
            [[0, 1, 2, 3], [], [0, 1]],
        ),
        (
            [(64, 128), (128, 128), (128, 10)],
            [True, True, True],
            [list(range(32)), list(range(48)), list(range(5))],
        ),
        (
            [(10, 20), (20, 30), (30, 40), (40, 5)],
            [False, True, False, True],
            [[], list(range(12)), [], [0, 1]],
        ),
        (
            [(4, 4)],
            [True],
            [[0, 1]],
        ),
    ]

    exact = 1.0
    ratio_score = 1.0

    for layer_shapes, depth_keep, width_keeps in cases:
        ref_count, ref_ratio = _oracle(layer_shapes, depth_keep, width_keeps)
        try:
            got_count, got_ratio = sol.compute_savings_from_chosen_width_depth(
                layer_shapes, depth_keep, width_keeps
            )
        except Exception:
            return {"size_ratio": 0.0, "exact_match": 0.0}

        if got_count != ref_count:
            exact = 0.0
        ratio_score = min(
            ratio_score,
            1.0 - abs(float(got_ratio) - float(ref_ratio)),
        )

    return {
        "size_ratio": max(0.0, ratio_score),
        "exact_match": exact,
    }
