"""Deterministic iteration-trace fixture: request-level (static) vs
iteration-level (continuous) batch scheduling.

Both a static-batching simulator (a batch is padded to its longest member
and swaps wholesale once every member is done) and a continuous-batching
simulator (a finished slot is immediately backfilled from the waiting
queue) are run over several request pools, so every run's ground-truth
label is known by construction from which simulator produced it.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def simulate_static(rng, num_requests, capacity, dur_range=(2, 6)):
    durations = rng.integers(dur_range[0], dur_range[1] + 1, size=num_requests)
    trace = []
    i = 0
    while i < num_requests:
        batch_ids = list(range(i, min(i + capacity, num_requests)))
        sync_dur = int(durations[i:i + capacity].max())
        for _ in range(sync_dur):
            trace.append(list(batch_ids))
        i += capacity
    return trace


def simulate_continuous(rng, num_requests, capacity, dur_range=(2, 6)):
    durations = rng.integers(dur_range[0], dur_range[1] + 1, size=num_requests)
    waiting = list(range(num_requests))
    active = {}
    trace = []
    while waiting or active:
        while len(active) < capacity and waiting:
            nid = waiting.pop(0)
            active[nid] = int(durations[nid])
        if not active:
            break
        trace.append(sorted(active.keys()))
        finished = [k for k in active if active[k] - 1 <= 0]
        for k in list(active.keys()):
            active[k] -= 1
        for k in finished:
            del active[k]
    return trace


def main() -> None:
    configs = [
        # (kind, num_requests, capacity, seed)
        ("static", 9, 3, 1),
        ("static", 6, 2, 11),
        ("static", 4, 4, 21),   # single batch, no swap at all
        ("static", 1, 3, 31),   # single request, trivial
        ("continuous", 9, 3, 2),
        ("continuous", 12, 4, 12),
        ("continuous", 7, 2, 22),
        ("continuous", 5, 3, 32),
        ("static", 10, 5, 41),
        ("continuous", 10, 5, 42),
        ("static", 8, 2, 51),
        ("continuous", 8, 2, 52),
    ]

    runs = []
    for kind, num_requests, capacity, seed in configs:
        rng = np.random.default_rng(seed)
        if kind == "static":
            trace = simulate_static(rng, num_requests, capacity)
        else:
            trace = simulate_continuous(rng, num_requests, capacity)
        runs.append((trace, num_requests))

    max_iters = max(len(t) for t, _ in runs)
    max_ids = max(n for _, n in runs)

    R = len(runs)
    active = np.zeros((R, max_iters, max_ids), dtype=np.int8)
    run_len = np.zeros((R,), dtype=np.int64)

    for r, (trace, num_requests) in enumerate(runs):
        run_len[r] = len(trace)
        for t, ids in enumerate(trace):
            for i in ids:
                active[r, t, i] = 1
        # pad remaining iteration slots by repeating the last real iteration
        # (irrelevant: check.py only ever reads active[r, :run_len[r], :])
        for t in range(len(trace), max_iters):
            active[r, t, :] = active[r, len(trace) - 1, :]

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "active.npy", active)
    np.save(out / "run_len.npy", run_len)


if __name__ == "__main__":
    main()
