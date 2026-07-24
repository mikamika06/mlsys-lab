"""Grader for `pyt-count-executed-value-stack-ops-in-a-hot-loop`.

Oracle for the op count: a real sys.settrace opcode-level tracer that reads
raw bytecode via frame.f_code.co_code and frame.f_lasti and names it with
dis.opname -- genuine CPython introspection, no simulation. Oracle for
correctness: numpy.polyval, independent of the candidate.
"""
from __future__ import annotations

import dis
import sys

import numpy as np


def _count_binary_ops(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) under opcode-level tracing and count every
    executed BINARY_OP instruction, across all frames entered during the
    call (so nested comprehensions/generators can't dodge the count)."""
    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if event == "call":
            frame.f_trace_opcodes = True
            return tracer
        if event == "opcode":
            op = frame.f_code.co_code[frame.f_lasti]
            if dis.opname[op] == "BINARY_OP":
                count += 1
        return tracer

    old = sys.settrace(tracer)
    try:
        result = fn(*args, **kwargs)
    finally:
        sys.settrace(old)
    return count, result


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    op_ratios = []
    rel_errs = []

    degrees = [0, 1, 2, 3, 5, 8]
    for n in degrees:
        coeffs = list(rng.uniform(-3.0, 3.0, size=n + 1))
        x = float(rng.uniform(-2.0, 2.0))

        try:
            op_count, got = _count_binary_ops(sol.horner_eval, coeffs, x)
        except Exception:
            return {"op_ratio": float("inf"), "rel_err": float("inf")}

        if not isinstance(got, (int, float)) or not np.isfinite(got):
            return {"op_ratio": float("inf"), "rel_err": float("inf")}

        target = 2 * n + 2  # Horner's exact 2n + small bookkeeping allowance
        op_ratios.append(op_count / target)

        expected = float(np.polyval(list(reversed(coeffs)), x))
        rel_errs.append(abs(got - expected) / (abs(expected) + 1e-12))

    return {
        "op_ratio": max(op_ratios),
        "rel_err": max(rel_errs),
    }
