import numpy as np
import sys
import inspect

def _reference_gather(arr, indices):
    return np.take(arr, indices)

def _count_line_events(func, *args):
    source_lines, first_lineno = inspect.getsourcelines(func)
    last_lineno = first_lineno + len(source_lines) - 1
    filename = func.__code__.co_filename

    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if event == 'line':
            lineno = frame.f_lineno
            if frame.f_code.co_filename == filename and first_lineno <= lineno <= last_lineno:
                count += 1
        return tracer

    sys.settrace(tracer)
    try:
        func(*args)
    finally:
        sys.settrace(None)
    return count

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)

    cases = [
        (rng.integers(0, 100, size=10), rng.choice(10, size=5, replace=True)),
        (rng.uniform(-1.0, 1.0, size=15), rng.choice(15, size=7, replace=False)),
        (np.array([], dtype=int), np.array([], dtype=int)),
    ]

    ok = 1.0
    max_events = 0

    for arr, indices in cases:
        try:
            ref_out = _reference_gather(arr, indices)
            # Capture line events
            got = sol.gather(arr, indices)  # may raise
            if not np.array_equal(got, ref_out):
                ok = 0.0
                break
            events = _count_line_events(sol.gather, arr, indices)
            if events > max_events:
                max_events = events
        except Exception:
            ok = 0.0
            break

    return {"exact_match": ok, "line_events": float(max_events)}
