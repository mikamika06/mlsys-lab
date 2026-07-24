"""Deterministic fixture of token-sequence insertion runs for the radix-tree
(compressed prefix tree) task.

Several hand-picked runs cover specific structural cases (basic split,
inserting an existing/ancestor prefix as a no-op, splitting exactly at an
edge boundary, fully disjoint branches, duplicate inserts, a three-way
split under an already-split node); several random-vocabulary runs give
broad coverage, since a small vocabulary forces frequent prefix sharing
and repeated splitting.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(2026)

    runs = []

    runs.append([[1, 2, 3, 4], [1, 2, 5, 6]])
    runs.append([[1, 2, 3, 4], [1, 2, 5, 6], [1, 2]])
    runs.append([[1, 2, 3, 4], [1, 2]])
    runs.append([[1, 2], [3, 4]])
    runs.append([[1, 2, 3], [1, 2, 3]])
    runs.append([[1, 2, 3], [9, 9]])
    runs.append([[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 3, 9]])
    runs.append([[5], [5, 6], [5, 6, 7], [5, 6, 7, 8]])  # nested prefixes, insert shortest->longest
    runs.append([[5, 6, 7, 8], [5, 6, 7], [5, 6], [5]])  # same set, longest->shortest

    for _ in range(6):
        vocab = 5
        n_seqs = int(rng.integers(4, 9))
        seqs = []
        for _ in range(n_seqs):
            length = int(rng.integers(1, 7))
            seqs.append([int(v) for v in rng.integers(0, vocab, size=length)])
        runs.append(seqs)

    all_seqs = []
    run_id = []
    for r, seqs in enumerate(runs):
        for s in seqs:
            all_seqs.append(s)
            run_id.append(r)

    max_len = max(len(s) for s in all_seqs)
    N = len(all_seqs)

    seqs_arr = np.full((N, max_len), -1, dtype=np.int64)
    seq_lens = np.zeros((N,), dtype=np.int64)
    for i, s in enumerate(all_seqs):
        seq_lens[i] = len(s)
        for j, tok in enumerate(s):
            seqs_arr[i, j] = tok

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "seqs.npy", seqs_arr)
    np.save(out / "seq_lens.npy", seq_lens)
    np.save(out / "run_id.npy", np.array(run_id, dtype=np.int64))


if __name__ == "__main__":
    main()
