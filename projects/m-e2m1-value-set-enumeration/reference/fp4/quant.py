from fp4.values import enumerate_values


def quantize_value(x):
    vals = enumerate_values()
    best_bits = 0
    best_diff = float("inf")
    for item in vals:
        diff = abs(item["value"] - x)
        if diff < best_diff:
            best_diff = diff
            best_bits = item["bits"]
    return best_bits
