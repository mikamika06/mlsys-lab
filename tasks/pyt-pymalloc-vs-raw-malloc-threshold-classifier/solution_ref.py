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
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            sizes.append(int(parts[1]))
    return sizes


def classify_pymalloc(sizes):
    if sysconfig.get_config_var("WITH_PYMALLOC") != 1:
        return [False for _ in sizes]
    classes = set(_debugmalloc_classes())
    return [(((int(size) + 7) // 8) * 8) in classes for size in sizes]
