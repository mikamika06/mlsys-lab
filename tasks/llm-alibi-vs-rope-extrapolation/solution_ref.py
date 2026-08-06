import math


def alibi_extrapolation_metric(num_heads, trained_len, extra_len):
    slopes = [2.0 ** (-(i + 1.0) / num_heads) for i in range(num_heads)]
    total = 0.0
    count = 0
    for q in range(trained_len, trained_len + extra_len):
        distances = [float(q - i) for i in range(q + 1)]
        for slope in slopes:
            logits = [-slope * d for d in distances]

            max_logit = logits[0]
            for val in logits[1:]:
                if val > max_logit:
                    max_logit = val

            logits_shifted = [val - max_logit for val in logits]
            weights = [math.exp(val) for val in logits_shifted]

            sum_weights = 0.0
            for w in weights:
                sum_weights += w

            weights = [w / sum_weights for w in weights]

            dot_prod = 0.0
            for w, d in zip(weights, distances):
                dot_prod += w * d

            total += dot_prod / (q + 1.0)
            count += 1

    return float(total / count)
