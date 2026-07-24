import sys
import numpy as np

def _count_lines(func, *args, **kwargs):
    counter = 0
    def tracer(frame, event, arg):
        nonlocal counter
        if event == 'line' and frame.f_code is func.__code__:
            counter += 1
        return tracer
    sys.settrace(tracer)
    try:
        result = func(*args, **kwargs)
    finally:
        sys.settrace(None)
    return result, counter

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        (rng.uniform(0.1, 10.0, size=10), rng.uniform(0.1, 10.0, size=10)),
        (rng.uniform(0.1, 5.0, size=20), rng.uniform(0.1, 5.0, size=20)),
        (rng.uniform(0.01, 2.0, size=50), rng.uniform(0.01, 2.0, size=50)),
        (rng.uniform(0.5, 8.0, size=100), rng.uniform(0.5, 8.0, size=100)),
        (rng.uniform(1.0, 12.0, size=200), rng.uniform(1.0, 12.0, size=200)),
    ]

    rel_err_acc = []
    op_counts = []

    for comp, comm in cases:
        try:
            got, op_count = _count_lines(sol.overlap_time, comp, comm)
        except Exception as e:
            return {"rel_err": 1.0, "op_count": 999}
        ref = max(np.sum(comp), np.sum(comm))
        rel_err = abs(got - ref) / (abs(ref) + 1e-12)
        rel_err_acc.append(rel_err)
        op_counts.append(op_count)

    overall_rel_err = max(rel_err_acc)
    overall_op_count = max(op_counts)
    return {"rel_err": overall_rel_err, "op_count": overall_op_count}
