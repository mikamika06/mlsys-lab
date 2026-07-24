"""Deterministic runtime probes for grading (hardware-independent counts)."""
from __future__ import annotations

import sys


def count_line_events(fn, *args, **kwargs) -> int:
    """Count Python-level line executions during ``fn(*args, **kwargs)``.

    A pure-Python loop over N elements emits ~N line events; a vectorised numpy
    call runs in C and emits almost none. So this is a robust, deterministic,
    hardware-independent discriminator between "looped" and "vectorised" solutions
    (numpy's C work is invisible to the tracer — only Python lines are counted).
    """
    n = 0

    def tracer(frame, event, arg):
        nonlocal n
        if event == "line":
            n += 1
        return tracer

    prev = sys.gettrace()
    sys.settrace(tracer)
    try:
        fn(*args, **kwargs)
    finally:
        sys.settrace(prev)
    return n
