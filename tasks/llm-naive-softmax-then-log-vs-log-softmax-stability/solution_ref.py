import math


def log_softmax(x: list[list[float]]) -> list[list[float]]:
    """Numerically stable log‑softmax along the last axis."""
    out = []
    for row in x:
        if not row:
            out.append([])
            continue
        mx = row[0]
        for val in row[1:]:
            if val > mx:
                mx = val

        s = 0.0
        for val in row:
            s += math.exp(val - mx)

        shift = mx + math.log(s)
        new_row = [val - shift for val in row]
        out.append(new_row)

    return out
