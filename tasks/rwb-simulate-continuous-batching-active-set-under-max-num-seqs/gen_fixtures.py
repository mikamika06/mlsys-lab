"""Deterministic fixture of request pools for the continuous-batching
active-set simulation task.

Several hand-picked runs cover specific scheduling scenarios (serial
contention, both requests fitting at once, an idle gap waiting for a late
arrival, staggered FIFO backfill, an arrival tie broken by original
index); several random runs give broad coverage.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(2026)

    runs = []
    # (arrival_iters, gen_lens, max_num_seqs)
    runs.append(([0, 0], [2, 2], 1))                       # serial contention
    runs.append(([0, 0], [2, 3], 2))                        # both fit at once
    runs.append(([0, 5], [1, 1], 1))                        # idle gap
    runs.append(([0, 0, 1], [2, 1, 3], 2))                  # staggered FIFO backfill
    runs.append(([2, 2, 2], [1, 1, 1], 1))                  # arrival tie, FIFO by index
    runs.append(([0], [5], 4))                              # single request, spare capacity
    runs.append(([0, 1, 2, 3, 4], [3, 3, 3, 3, 3], 2))       # steady arrivals, tight cap

    for _ in range(6):
        n = int(rng.integers(4, 9))
        max_arrival = int(rng.integers(0, 6))
        arrival_iters = rng.integers(0, max_arrival + 1, size=n).tolist()
        gen_lens = rng.integers(1, 6, size=n).tolist()
        cap = int(rng.integers(1, 4))
        runs.append((arrival_iters, gen_lens, cap))

    all_arrival = []
    all_gen = []
    run_id = []
    caps = []
    for r, (arrival_iters, gen_lens, cap) in enumerate(runs):
        for a, g in zip(arrival_iters, gen_lens):
            all_arrival.append(a)
            all_gen.append(g)
            run_id.append(r)
        caps.append(cap)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "arrival_iters.npy", np.array(all_arrival, dtype=np.int64))
    np.save(out / "gen_lens.npy", np.array(all_gen, dtype=np.int64))
    np.save(out / "run_id.npy", np.array(run_id, dtype=np.int64))
    np.save(out / "max_num_seqs.npy", np.array(caps, dtype=np.int64))


if __name__ == "__main__":
    main()
