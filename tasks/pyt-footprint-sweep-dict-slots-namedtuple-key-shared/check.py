import sys
from collections import namedtuple

import numpy as np


def _measure(width):
    names = [f"f{i}" for i in range(width)]

    class DictBacked:
        pass

    d = DictBacked()
    for i, name in enumerate(names):
        setattr(d, name, i)
    dict_size = sys.getsizeof(d) + sys.getsizeof(d.__dict__)

    SlotType = type("SlotType", (), {"__slots__": tuple(names)})
    s = SlotType()
    for i, name in enumerate(names):
        setattr(s, name, i)
    slot_size = sys.getsizeof(s)

    NT = namedtuple("NT", names)
    nt = NT(*range(width))
    namedtuple_size = sys.getsizeof(nt)

    class Shared:
        def __init__(self):
            for j, name in enumerate(names):
                setattr(self, name, j)

    a = Shared()
    b = Shared()
    shared_size = sys.getsizeof(b) + sys.getsizeof(b.__dict__)

    # A second instance ensures CPython has created and reused the shared key table.
    _ = a
    return np.array(
        [
            dict_size / slot_size,
            namedtuple_size / slot_size,
            shared_size / slot_size,
        ],
        dtype=np.float64,
    )


def _oracle(widths):
    return np.vstack([_measure(int(w)) for w in widths]).astype(np.float64)


def grade(sol, fx) -> dict:
    widths = [1, 2, 4, 8, 16, 32]
    ref = _oracle(widths)
    try:
        got = np.asarray(sol.footprint_sweep(widths), dtype=np.float64)
    except Exception:
        return {"size_ratio": 0.0}
    if got.shape != ref.shape:
        return {"size_ratio": 0.0}
    dev = np.max(np.abs(got - ref) / (np.abs(ref) + 1e-12))
    return {"size_ratio": float(1.0 / (1.0 + dev))}
