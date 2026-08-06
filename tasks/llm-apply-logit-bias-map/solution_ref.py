def apply_logit_bias_map(logits: list[list[float]], bias_map: dict[int, float]) -> list[list[float]]:
    """
    Correct implementation that adds the bias values to every row of logits.
    """
    n = len(logits)
    d = len(logits[0]) if n > 0 else 0
    bias = [0.0] * d
    for token, value in bias_map.items():
        if 0 <= token < d:
            bias[token] += value

    out = []
    for row in logits:
        new_row = [x + b for x, b in zip(row, bias)]
        out.append(new_row)
    return out
