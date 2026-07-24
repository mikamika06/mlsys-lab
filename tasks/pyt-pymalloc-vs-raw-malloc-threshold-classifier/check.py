import os
import sys
import sysconfig


def _debugmalloc_classes():
    if not hasattr(sys, "_debugmallocstats"):
        return []
    r, w = os.pipe()
    old = os.dup(2)
    try:
        os.dup2(w, 2)
        sys._debugmallocstats()
    finally:
        os.close(w)
        os.dup2(old, 2)
        os.close(old)
    text = os.read(r, 1_000_000).decode(errors="ignore")
    sizes = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            if parts[0].isdigit() and parts[1].isdigit():
                sizes.append(int(parts[1]))
    return sizes


def _oracle(size):
    if sysconfig.get_config_var("WITH_PYMALLOC") != 1:
        return False
    classes = _debugmalloc_classes()
    if not classes:
        return False
    rounded = ((int(size) + 7) // 8) * 8
    return rounded in classes


def grade(sol, fx) -> dict:
    sizes = [480, 488, 496, 504, 512, 513, 520, 528, 600]
    ref = [_oracle(s) for s in sizes]
    try:
        got = list(sol.classify_pymalloc(sizes))
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == ref else 0.0}
