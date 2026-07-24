import numpy as np


def _ref_order(ops):
    """The real oracle: apply ops to an actual CPython dict."""
    d = {}
    for op, k in ops:
        if op == "set":
            d[k] = 0
        elif op == "del":
            del d[k]
        else:
            raise ValueError(f"bad op {op!r}")
    return list(d.keys())


def _gen_ops(rng, n_ops, key_universe):
    present = set()
    ops = []
    for _ in range(n_ops):
        if not present or rng.random() < 0.6:
            k = int(rng.integers(0, key_universe))
            ops.append(("set", k))
            present.add(k)
        else:
            k = int(rng.choice(sorted(present)))
            ops.append(("del", k))
            present.discard(k)
    return ops


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    cases = [
        [("set", 1), ("set", 2), ("set", 3), ("del", 2), ("set", 2)],
        [("set", 1), ("set", 2), ("set", 3), ("set", 1)],
        [("set", 5), ("set", 5), ("del", 5), ("set", 5), ("set", 6), ("set", 5)],
        [("set", 0), ("set", 1), ("set", 2), ("set", 3), ("set", 4),
         ("del", 0), ("del", 4), ("set", 0), ("set", 2), ("set", 4)],
    ]
    for n_ops, key_universe in [(20, 6), (50, 10), (200, 15), (400, 25), (60, 4)]:
        cases.append(_gen_ops(rng, n_ops, key_universe))

    exact = 1.0
    for ops in cases:
        ref = _ref_order(ops)
        try:
            got = sol.predict_dict_order(list(ops))
        except Exception:
            exact = 0.0
            break
        try:
            got = list(got)
        except Exception:
            exact = 0.0
            break
        if got != ref:
            exact = 0.0
            break

    return {"exact_match": exact}
