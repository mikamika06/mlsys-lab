"""Grader for `pyt-cap-peak-live-large-object-count`.

Oracle for peak live count: a real class whose __init__/__del__ update a
live-instance counter -- genuine CPython reference-counting behaviour, not
a simulation. Oracle for the correctness gate: a plain arithmetic sum,
independent of the candidate's `process` and of the tracked class.
"""
from __future__ import annotations

TARGET_PEAK = 2


class _TrackedBuffer:
    live = 0
    peak = 0

    def __init__(self, size):
        self.size = size
        self.data = list(range(size))
        _TrackedBuffer.live += 1
        if _TrackedBuffer.live > _TrackedBuffer.peak:
            _TrackedBuffer.peak = _TrackedBuffer.live

    def checksum(self) -> float:
        return float(sum(self.data))

    def __del__(self):
        _TrackedBuffer.live -= 1


def _reference_total(sizes) -> float:
    return sum(s * (s - 1) / 2.0 for s in sizes)


def grade(sol, fx) -> dict:
    size_lists = [
        [50 + i for i in range(8)],
        [30 + 3 * i for i in range(20)],
        [100] * 15,
        [10, 500, 20, 300, 10, 10, 10, 10],
    ]

    peaks = []
    total_errs = []

    for sizes in size_lists:
        _TrackedBuffer.live = 0
        _TrackedBuffer.peak = 0

        try:
            got_total = sol.process(list(sizes), _TrackedBuffer)
        except Exception:
            return {"peak_ratio": float("inf"), "total_rel_err": float("inf")}

        if not isinstance(got_total, (int, float)):
            return {"peak_ratio": float("inf"), "total_rel_err": float("inf")}

        expected_total = _reference_total(sizes)
        denom = abs(expected_total) + 1e-12
        total_errs.append(abs(got_total - expected_total) / denom)
        peaks.append(_TrackedBuffer.peak)

    return {
        "peak_ratio": max(peaks) / TARGET_PEAK,
        "total_rel_err": max(total_errs),
    }
