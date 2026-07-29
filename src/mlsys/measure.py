from __future__ import annotations

import statistics
import time
from typing import Callable


def _percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


def bench(fn: Callable[[], object], warmup: int = 3, reps: int = 9,
          sync: Callable[[], None] | None = None) -> dict:
    for _ in range(max(0, warmup)):
        fn()
    if sync:
        sync()
    samples = []
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append(time.perf_counter() - t0)
    q1, q3 = _percentile(samples, 0.25), _percentile(samples, 0.75)
    med = statistics.median(samples)
    return {"median": med, "iqr": q3 - q1, "q1": q1, "q3": q3,
            "min": min(samples), "reps": len(samples),
            "rel_iqr": (q3 - q1) / med if med else float("inf")}


def ab(fa: Callable[[], object], fb: Callable[[], object], rounds: int = 7,
       warmup: int = 2, sync: Callable[[], None] | None = None) -> dict:
    """Interleaved A/B.

    Measuring all of A and then all of B attributes thermal drift to B: on a laptop
    the second half of a long run is slower for reasons that have nothing to do with
    the code. Alternating the two shares the drift between them, so the ratio stays
    a property of the code rather than of the machine's temperature.

    `separable` is 1 only when the two interquartile ranges do not overlap — a
    speed-up you cannot separate from the noise is not a speed-up you may gate on.
    """
    for _ in range(max(0, warmup)):
        fa()
        fb()
    if sync:
        sync()
    a_s: list[float] = []
    b_s: list[float] = []
    for i in range(max(1, rounds)):
        first, second, fs, ss = (fa, fb, a_s, b_s) if i % 2 == 0 else (fb, fa, b_s, a_s)
        t0 = time.perf_counter()
        first()
        if sync:
            sync()
        fs.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        second()
        if sync:
            sync()
        ss.append(time.perf_counter() - t0)

    a_med, b_med = statistics.median(a_s), statistics.median(b_s)
    a_q1, a_q3 = _percentile(a_s, 0.25), _percentile(a_s, 0.75)
    b_q1, b_q3 = _percentile(b_s, 0.25), _percentile(b_s, 0.75)
    separable = 1.0 if (a_q3 < b_q1 or b_q3 < a_q1) else 0.0
    return {
        "a_median": a_med, "b_median": b_med,
        "a_iqr": a_q3 - a_q1, "b_iqr": b_q3 - b_q1,
        "ratio": (a_med / b_med) if b_med else float("inf"),
        "separable": separable,
        "rounds": len(a_s),
    }


def reps_for(sample: list[float], half_width_rel: float = 0.05, z: float = 1.96) -> int:
    """How many repetitions a target confidence half-width needs, from a pilot sample."""
    if len(sample) < 2:
        return 30
    m = statistics.mean(sample)
    s = statistics.stdev(sample)
    if m == 0:
        return 30
    return max(3, int((z * s / (half_width_rel * m)) ** 2 + 0.999))


def torch_sync():
    """A sync callable for whichever accelerator torch is actually using, or None."""
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return torch.cuda.synchronize
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.mps.synchronize
    return None
