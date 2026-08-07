import numpy as np


def allocate_bits(sensitivities, bit_options, total_budget_bits):
    n = len(sensitivities)
    min_bit = min(bit_options)
    max_bit = max(bit_options)
    bits = [min_bit] * n
    current_bits = sum(bits)

    while current_bits < total_budget_bits:
        gains = []
        for i in range(n):
            current_b = bits[i]
            higher_options = [b for b in bit_options if b > current_b]
            if not higher_options or current_bits + (min(higher_options) - current_b) > total_budget_bits:
                gains.append(-float("inf"))
            else:
                next_b = min(higher_options)
                gains.append(sensitivities[i] * (next_b - current_b))

        if all(g == -float("inf") for g in gains):
            break

        best_i = int(np.argmax(gains))
        higher_options = [b for b in bit_options if b > bits[best_i]]
        next_b = min(higher_options)
        current_bits += (next_b - bits[best_i])
        bits[best_i] = next_b

    return bits
