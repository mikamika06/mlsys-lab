"""Deterministic fixture of valid token-prefixes from a JSON-subset grammar,
for the pushdown-automaton next-token-mask task.

Token vocabulary (index order used for the .npy encoding):
    0:'{'  1:'}'  2:'['  3:']'  4:':'  5:','
    6:'STR' 7:'NUM' 8:'TRUE' 9:'FALSE' 10:'NULL'

Random complete documents are generated from the grammar

    value  := STR | NUM | TRUE | FALSE | NULL | object | array
    object := '{' '}' | '{' STR ':' value (',' STR ':' value)* '}'
    array  := '[' ']' | '[' value (',' value)* ']'

and every prefix of every document (including the empty prefix and the full
document) becomes one test example, padded to a common length with -1.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np

TOKENS = ["{", "}", "[", "]", ":", ",", "STR", "NUM", "TRUE", "FALSE", "NULL"]
TOKEN_ID = {t: i for i, t in enumerate(TOKENS)}


def _gen_value(rng, depth, max_depth):
    if depth >= max_depth:
        choices = ["STR", "NUM", "TRUE", "FALSE", "NULL"]
    else:
        choices = ["STR", "NUM", "TRUE", "FALSE", "NULL", "OBJ", "ARR"]
    kind = choices[rng.integers(len(choices))]
    if kind == "OBJ":
        return _gen_object(rng, depth + 1, max_depth)
    if kind == "ARR":
        return _gen_array(rng, depth + 1, max_depth)
    return [kind]


def _gen_object(rng, depth, max_depth):
    n_members = int(rng.integers(0, 4))
    toks = ["{"]
    for i in range(n_members):
        if i > 0:
            toks.append(",")
        toks.append("STR")
        toks.append(":")
        toks += _gen_value(rng, depth, max_depth)
    toks.append("}")
    return toks


def _gen_array(rng, depth, max_depth):
    n_elems = int(rng.integers(0, 4))
    toks = ["["]
    for i in range(n_elems):
        if i > 0:
            toks.append(",")
        toks += _gen_value(rng, depth, max_depth)
    toks.append("]")
    return toks


def main() -> None:
    rng = np.random.default_rng(2026)
    docs = [_gen_value(rng, 0, 3) for _ in range(40)]

    examples = []
    for doc in docs:
        for L in range(len(doc) + 1):
            examples.append(doc[:L])

    max_len = max((len(e) for e in examples), default=0)
    N = len(examples)

    prefixes = np.full((N, max_len), -1, dtype=np.int8)
    lengths = np.zeros((N,), dtype=np.int64)
    for i, ex in enumerate(examples):
        lengths[i] = len(ex)
        for j, tok in enumerate(ex):
            prefixes[i, j] = TOKEN_ID[tok]

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "prefixes.npy", prefixes)
    np.save(out / "lengths.npy", lengths)


if __name__ == "__main__":
    main()
