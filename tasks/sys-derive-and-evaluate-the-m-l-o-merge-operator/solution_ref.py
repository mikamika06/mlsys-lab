from __future__ import annotations

import math


def merge_mlo(
    state1: tuple[float, float, list[float]],
    state2: tuple[float, float, list[float]],
) -> tuple[float, float, list[float]]:
    m1, l1, o1 = state1
    m2, l2, o2 = state2

    mf1 = float(m1)
    mf2 = float(m2)
    m = mf1 if mf1 > mf2 else mf2
    a1 = math.exp(mf1 - m)
    a2 = math.exp(mf2 - m)

    l = float(l1) * a1 + float(l2) * a2

    o = []
    for i in range(len(o1)):
        o.append(float(o1[i]) * a1 + float(o2[i]) * a2)

    return m, l, o
