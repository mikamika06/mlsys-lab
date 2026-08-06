def mask_banned_tokens(logits: list[float] | list[list[float]], banned_indices: list[int]) -> list[float] | list[list[float]]:
    if not logits:
        return []
    if isinstance(logits[0], list):
        out = []
        for row in logits:
            new_row = list(row)
            for idx in banned_indices:
                if 0 <= idx < len(new_row):
                    new_row[idx] = float('-inf')
            out.append(new_row)
        return out
    else:
        out = list(logits)
        for idx in banned_indices:
            if 0 <= idx < len(out):
                out[idx] = float('-inf')
        return out
