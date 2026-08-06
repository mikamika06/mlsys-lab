def apply_repetition_penalty(logits: list[float], penalty: float) -> list[float]:
    out = []
    for x in logits:
        if x > 0:
            out.append(x / penalty)
        elif x < 0:
            out.append(x * penalty)
        else:
            out.append(0.0)
    return out
