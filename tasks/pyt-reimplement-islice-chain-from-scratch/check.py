"""Grader for `pyt-reimplement-islice-chain-from-scratch`.

Oracle for correctness: the real itertools.islice / itertools.chain (real
CPython stdlib, independent of the candidate). Oracle for the "genuinely
iterated element by element" requirement: a sys.settrace line-event
tracer (same technique as arena.probe.count_line_events) applied while
draining the candidate's generator with collections.deque(gen, maxlen=0)
-- a pure-C drain, so all measured events come from the candidate's own
Python-level implementation, not the consuming loop.
"""
from __future__ import annotations

import itertools
import sys
from collections import deque


def _count_line_events(fn, *args, **kwargs) -> int:
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


def _drain(gen) -> None:
    deque(gen, maxlen=0)


def _islice_cases():
    # (source_factory, start, stop, step, traversal_floor)
    return [
        (lambda: iter(range(100)), 5, 50, 2, 50),
        (lambda: iter(range(200)), 0, 90, 1, 90),
        (lambda: iter(list(range(60))), 10, 60, 3, 60),
        (lambda: (x * x for x in range(80)), 0, 20, 1, 20),
        (lambda: iter(range(30)), 25, 30, 1, 30),
        (lambda: iter(range(10)), 3, 3, 1, 3),   # empty output, nonzero skip
        (lambda: iter(range(15)), 0, 100, 1, 15),  # stop beyond source length
    ]


def _chain_cases():
    # (iterables_factory, traversal_floor)
    return [
        (lambda: ([1, 2, 3], (4, 5, 6, 7), range(10)), 3 + 4 + 10),
        (lambda: (range(0), [1], (x for x in range(40))), 0 + 1 + 40),
        (lambda: (list(range(25)),), 25),
        (lambda: ([], [], []), 0),
        (lambda: (range(20), range(20), range(20)), 60),
    ]


def grade(sol, fx) -> dict:
    islice_matches = []
    islice_ratios = []
    for src_factory, start, stop, step, floor in _islice_cases():
        expected = list(itertools.islice(src_factory(), start, stop, step))
        try:
            gen = sol.my_islice(src_factory(), start, stop, step)
            got = list(gen)
        except Exception:
            islice_matches.append(0.0)
            islice_ratios.append(0.0)
            continue
        islice_matches.append(1.0 if got == expected else 0.0)

        try:
            events = _count_line_events(_drain, sol.my_islice(src_factory(), start, stop, step))
        except Exception:
            events = 0
        islice_ratios.append(events / max(floor, 1))

    chain_matches = []
    chain_ratios = []
    for iterables_factory, floor in _chain_cases():
        expected = list(itertools.chain(*iterables_factory()))
        try:
            gen = sol.my_chain(*iterables_factory())
            got = list(gen)
        except Exception:
            chain_matches.append(0.0)
            chain_ratios.append(0.0)
            continue
        chain_matches.append(1.0 if got == expected else 0.0)

        try:
            events = _count_line_events(_drain, sol.my_chain(*iterables_factory()))
        except Exception:
            events = 0
        chain_ratios.append(events / max(floor, 1))

    return {
        "islice_exact_match": min(islice_matches) if islice_matches else 0.0,
        "chain_exact_match": min(chain_matches) if chain_matches else 0.0,
        "islice_event_ratio": min(islice_ratios) if islice_ratios else 0.0,
        "chain_event_ratio": min(chain_ratios) if chain_ratios else 0.0,
    }
