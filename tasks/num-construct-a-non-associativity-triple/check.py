import numpy as np


def _bits(x):
    return np.asarray(np.float32(x), dtype=np.float32).view(np.uint32).item()


def _oracle_has_nonassoc():
    values = [np.float32(0.0), np.float32(-0.0)]
    for e in range(-20, 21):
        v = np.float32(2.0 ** e)
        values.append(v)
        values.append(np.float32(-v))

    for a in values:
        for b in values:
            for c in values:
                left = np.float32(np.float32(a + b) + c)
                right = np.float32(a + np.float32(b + c))
                if _bits(left) != _bits(right):
                    return True
    return False


def grade(sol, fx) -> dict:
    oracle = _oracle_has_nonassoc()
    ok = 0.0
    if oracle:
        try:
            a, b, c = sol.construct_nonassoc_triple()
            a = np.float32(a)
            b = np.float32(b)
            c = np.float32(c)
            left = np.float32(np.float32(a + b) + c)
            right = np.float32(a + np.float32(b + c))
            if _bits(left) != _bits(right):
                ok = 1.0
        except Exception:
            ok = 0.0
    return {"exact_match": ok}
