import numpy as np
from mlsys import scorers


def _oracle_encode(text, vocab, merges):
    symbols = [bytes([b]) for b in text.encode("utf-8")]

    while True:
        best = None
        best_rank = None
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            if pair in merges:
                rank = merges[pair]
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best = pair
        if best is None:
            break

        merged = best[0] + best[1]
        out = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == best[0] and symbols[i + 1] == best[1]:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out

    return [vocab[s] for s in symbols]


def grade(sol, fx) -> dict:
    cases = [
        (
            "hello",
            {
                b"h": 1, b"e": 2, b"l": 3, b"o": 4,
                b"he": 10, b"ll": 11, b"llo": 12
            },
            {
                (b"h", b"e"): 0,
                (b"l", b"l"): 1,
                (b"ll", b"o"): 2
            },
        ),
        (
            "hi🙂",
            {
                b"h": 1, b"i": 2,
                b"\xf0": 3, b"\x9f": 4, b"\x99": 5, b"\x82": 6,
                b"\xf0\x9f\x99\x82": 20,
                b"hi": 30,
            },
            {
                (b"h", b"i"): 0,
                (b"\xf0", b"\x9f"): 1,
                (b"\xf0\x9f", b"\x99"): 2,
                (b"\xf0\x9f\x99", b"\x82"): 3,
            },
        ),
        (
            "café",
            {
                b"c": 1, b"a": 2, b"f": 3,
                b"\xc3": 4, b"\xa9": 5,
                b"\xc3\xa9": 40,
            },
            {
                (b"\xc3", b"\xa9"): 0,
            },
        ),
    ]

    score = 1.0
    for text, vocab, merges in cases:
        ref = _oracle_encode(text, vocab, merges)
        try:
            got = sol.byte_bpe_encode(text, vocab, merges)
        except Exception:
            score = 0.0
            break
        score = min(
            score,
            scorers.byte_exact_fraction(
                np.asarray(ref, dtype=np.int32),
                np.asarray(got, dtype=np.int32),
            ),
        )
    return {"byte_exact_fraction": score}
