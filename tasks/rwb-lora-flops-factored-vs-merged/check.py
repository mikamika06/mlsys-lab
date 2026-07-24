def _ref(d, r, max_sequence_length):
    factored = 2 * d * r + 2 * r * d
    merged = 2 * d * d
    merge = 2 * d * d * r

    break_even = None
    denominator = factored - merged
    if denominator > 0:
        s = (merge + denominator - 1) // denominator
        if 1 <= s <= max_sequence_length:
            break_even = s

    return {
        "factored_flops_per_token": factored,
        "merged_flops_per_token": merged,
        "merge_flops": merge,
        "break_even_sequence_length": break_even,
    }


def grade(sol, fx) -> dict:
    cases = [
        (4096, 8, 1000),
        (1024, 64, 10000),
        (128, 4, 200),
        (512, 512, 100),
        (256, 1, 10000),
        (64, 8, 100),
    ]

    ok = 1.0
    for d, r, max_len in cases:
        try:
            got = sol.lora_break_even(d, r, max_len)
        except Exception:
            ok = 0.0
            break
        if got != _ref(d, r, max_len):
            ok = 0.0
            break

    return {"exact_match": ok}
