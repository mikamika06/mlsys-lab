import numpy as np


def _merge(a, b):
    m1, s1 = a
    m2, s2 = b
    m = max(m1, m2)
    return (
        m,
        s1 * np.exp(m1 - m) + s2 * np.exp(m2 - m),
    )


def _left(parts):
    out = parts[0]
    for part in parts[1:]:
        out = _merge(out, part)
    return out


def _right(parts):
    if len(parts) == 1:
        return parts[0]
    return _merge(parts[0], _right(parts[1:]))


def _check(parts):
    vals = [_left(parts), _right(parts)]
    if len(parts) >= 3:
        vals.append(_merge(_merge(parts[0], parts[1]), _merge(*parts[2:4]) if len(parts) > 3 else parts[2]))
    a = vals[0]
    return all(abs(a[0] - b[0]) <= 1e-6 and abs(a[1] - b[1]) <= 1e-6 for b in vals[1:])


def check_block_merge_associativity(rows):
    rows = np.asarray(rows, dtype=np.float64)
    answer = []
    for row in rows:
        valid = True
        for count in range(2, 6):
            cuts = np.linspace(0, len(row), count + 1, dtype=int)
            blocks = []
            for i in range(count):
                block = row[cuts[i]:cuts[i + 1]]
                m = np.max(block)
                s = np.sum(np.exp(block - m))
                blocks.append((float(m), float(s)))
            valid = valid and _check(blocks)
        answer.append(valid)
    return np.asarray(answer, dtype=bool)
