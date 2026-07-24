import sys
from collections import namedtuple

import numpy as np


def footprint_sweep(widths):
    rows = []
    for width in widths:
        width = int(width)
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

        first = Shared()
        second = Shared()
        _ = first
        shared_size = sys.getsizeof(second) + sys.getsizeof(second.__dict__)

        rows.append(
            [
                dict_size / slot_size,
                namedtuple_size / slot_size,
                shared_size / slot_size,
            ]
        )

    return np.asarray(rows, dtype=np.float64)
