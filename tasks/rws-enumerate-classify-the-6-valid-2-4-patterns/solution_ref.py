def _build_reference():
    patterns = [tuple(int(b) for b in format(i, '04b')) for i in range(16)]
    valid = sorted([p for p in patterns if sum(p) == 2])
    return {p: idx for idx, p in enumerate(valid)}

_REFERENCE_MAP = _build_reference()

def classify_patterns(vectors: list[list[int]]) -> list[int]:
    result = []
    for row in vectors:
        row_sum = 0
        for val in row:
            row_sum += val

        idx = -1
        if row_sum == 2:
            t = tuple(row)
            if t in _REFERENCE_MAP:
                idx = _REFERENCE_MAP[t]
        result.append(idx)

    return result
