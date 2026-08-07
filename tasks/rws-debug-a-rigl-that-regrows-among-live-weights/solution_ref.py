def rigl_grow(
    mask: list[int],
    weights: list[float],
    grads: list[float],
    grow_count: int,
) -> list[int]:
    del weights
    out = list(mask)
    zero_indices = [i for i, val in enumerate(out) if val == 0]
    count = min(int(grow_count), len(zero_indices))
    if count:
        candidates = sorted(zero_indices, key=lambda i: abs(grads[i]), reverse=True)
        chosen = candidates[:count]
        for idx in chosen:
            out[idx] = 1
    return out
