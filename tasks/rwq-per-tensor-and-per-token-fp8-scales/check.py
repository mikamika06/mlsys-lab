import math
import random


def _ref_fp8_scales(W: list[list[float]], X: list) -> tuple[float, list[float]]:
    max_w = -float('inf')
    for row in W:
        for val in row:
            abs_val = abs(val)
            if abs_val > max_w:
                max_w = abs_val
    tensor_scale = max_w / 448.0

    def get_tokens(tensor):
        if not isinstance(tensor, list):
            return []
        if len(tensor) == 0:
            return []
        if not isinstance(tensor[0], list):
            return [tensor]
        if not isinstance(tensor[0][0], list):
            return tensor
        tokens = []
        for sub in tensor:
            tokens.extend(get_tokens(sub))
        return tokens

    tokens = get_tokens(X)

    token_scales = []
    for token in tokens:
        max_x = -float('inf')
        for val in token:
            abs_val = abs(val)
            if abs_val > max_x:
                max_x = abs_val
        token_scales.append(max_x / 448.0)

    return tensor_scale, token_scales


def _calc_rel_err(ref_tensor_scale, ref_token_scales, cand_tensor_scale, cand_token_scales):
    ref_vec = [ref_tensor_scale] + list(ref_token_scales)
    cand_vec = [cand_tensor_scale] + list(cand_token_scales)

    if len(ref_vec) != len(cand_vec):
        return float("inf")

    sq_diff = sum((c - r) ** 2 for c, r in zip(cand_vec, ref_vec))
    sq_ref = sum(r ** 2 for r in ref_vec)

    if sq_ref == 0.0:
        return 0.0 if sq_diff == 0.0 else float("inf")

    return math.sqrt(sq_diff) / math.sqrt(sq_ref)


def _gen_nested_list(shape, rng):
    if len(shape) == 1:
        return [rng.uniform(-10.0, 10.0) for _ in range(shape[0])]
    return [_gen_nested_list(shape[1:], rng) for _ in range(shape[0])]


def grade(sol, fx) -> dict:
    rng = random.Random(42)

    test_cases = [
        (
            [[0.0, -3.0], [4.0, 1.0]],
            [[2.0, -5.0], [-1.0, 7.0]],
        ),
        (
            _gen_nested_list([4, 8], rng),
            _gen_nested_list([16, 8], rng),
        ),
        (
            _gen_nested_list([5, 10], rng),
            _gen_nested_list([3, 4, 10], rng),
        ),
        (
            _gen_nested_list([3, 6], rng),
            _gen_nested_list([2, 3, 2, 6], rng),
        ),
        (
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ),
        (
            [[-12.5]],
            [[3.2, -448.0, 100.0]],
        ),
    ]

    max_rel_err = 0.0

    for W, X in test_cases:
        ref_tensor_scale, ref_token_scales = _ref_fp8_scales(W, X)
        cand_tensor_scale, cand_token_scales = sol.fp8_scales(W, X)

        err = _calc_rel_err(ref_tensor_scale, ref_token_scales, cand_tensor_scale, cand_token_scales)
        if err > max_rel_err:
            max_rel_err = err

    return {"rel_err": max_rel_err}
