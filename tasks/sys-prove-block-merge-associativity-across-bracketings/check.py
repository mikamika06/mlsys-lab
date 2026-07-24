import numpy as np


def _merge(a, b):
    m1, s1 = a
    m2, s2 = b
    m = max(m1, m2)
    s = s1 * np.exp(m1 - m) + s2 * np.exp(m2 - m)
    return float(m), float(s)


def _summary(block):
    m = float(np.max(block))
    s = float(np.sum(np.exp(block - m)))
    return m, s


def _left_reduce(parts):
    out = parts[0]
    for p in parts[1:]:
        out = _merge(out, p)
    return out


def _right_reduce(parts):
    if len(parts) == 1:
        return parts[0]
    return _merge(parts[0], _right_reduce(parts[1:]))


def _oracle_row(row, split_count):
    n = len(row)
    cuts = np.linspace(0, n, split_count + 1, dtype=int)
    parts = [_summary(row[cuts[i]:cuts[i + 1]]) for i in range(split_count)]
    results = [
        _left_reduce(parts),
        _right_reduce(parts),
    ]
    if len(parts) >= 3:
        results.append(_merge(_merge(parts[0], parts[1]), _merge(parts[2], parts[3]) if len(parts) > 3 else parts[2]))
    base = results[0]
    for item in results[1:]:
        if abs(base[0] - item[0]) > 1e-6 or abs(base[1] - item[1]) > 1e-6:
            return False
    return True


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    rows = rng.normal(size=(12, 20)).astype(np.float64)
    rows[0, :5] = np.array([1000, 999, 998, 997, 996], dtype=np.float64)
    expected = []
    for row in rows:
        ok = True
        for blocks in range(2, 6):
            if not _oracle_row(row, blocks):
                ok = False
        expected.append(ok)
    expected = np.array(expected, dtype=bool)
    try:
        got = np.asarray(sol.check_block_merge_associativity(rows), dtype=bool)
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": float(np.array_equal(got, expected))}
