"""Deterministic fixture: candidate binary strings to test against the
grammar "binary encoding of a non-negative integer divisible by 3"
(the empty string encodes 0, which is divisible by 3).

Run with:
    python3 tasks/rwb-does-the-string-parse-against-the-grammar/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(13)
    strings = ['']
    for _ in range(40):
        n = int(rng.integers(1, 12))
        s = ''.join(rng.choice(['0', '1'], size=n))
        strings.append(s)
    # a few hand-picked edge cases
    strings += ['0', '1', '00', '11', '011', '0110', '1001100', '111111111']
    return np.array(strings, dtype='<U16')


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    strings = build()
    np.save(OUT / "strings.npy", strings)
    print("wrote", strings.shape)
