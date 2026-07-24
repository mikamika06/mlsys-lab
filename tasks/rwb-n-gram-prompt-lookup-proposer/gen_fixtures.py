"""Deterministic token-sequence fixture for the n-gram / prompt-lookup
speculative-decoding proposer task.

A small-vocabulary random stream is stitched together with several
deliberately repeated templates (so genuine longest-suffix matches exist at
multiple points, at multiple lengths), leaving plenty of positions where no
match exists either.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(2026)
    vocab = 8

    def rand_run(n):
        return list(rng.integers(0, vocab, size=n))

    seq = []
    seq += rand_run(12)
    template_a = [1, 2, 3, 4, 5]
    seq += template_a
    seq += rand_run(6)
    template_b = [6, 0, 6, 0]
    seq += template_b
    seq += rand_run(8)
    seq += template_a          # repeats template_a -> long-ngram match available
    seq += rand_run(5)
    seq += template_b[:2]      # partial repeat -> only a short-ngram match
    seq += rand_run(10)
    seq += template_a[:3]      # another partial repeat of template_a's prefix
    seq += rand_run(4)
    seq += [7, 7]              # trailing pair unlikely to recur -> no-match case

    sequence = np.array(seq, dtype=np.int64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "sequence.npy", sequence)


if __name__ == "__main__":
    main()
