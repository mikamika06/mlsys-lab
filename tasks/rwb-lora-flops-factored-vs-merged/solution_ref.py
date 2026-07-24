def lora_break_even(d, r, max_sequence_length):
    factored = 2 * d * r + 2 * r * d
    merged = 2 * d * d
    merge = 2 * d * d * r

    break_even = None
    denominator = factored - merged
    if denominator > 0:
        length = (merge + denominator - 1) // denominator
        if length <= max_sequence_length:
            break_even = length

    return {
        "factored_flops_per_token": factored,
        "merged_flops_per_token": merged,
        "merge_flops": merge,
        "break_even_sequence_length": break_even,
    }
