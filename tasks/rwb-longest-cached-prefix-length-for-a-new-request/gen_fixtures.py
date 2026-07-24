"""Deterministic prefix-cache fixture built with a chained rolling hash.

cached_hashes holds the block hashes of three "other" cached requests:
  - seq_A shares its first two blocks EXACTLY with the new request, then
    diverges at block 2.
  - seq_B is unrelated noise.
  - seq_C is a DECOY: it contains a block whose raw CONTENT is identical
    to the new request's (missing) block 2, but at a different chain
    position, so its true chain hash does not equal what the new
    request's block 2 needs. A naive "hash the block content only,
    ignore the parent" implementation would wrongly treat this as a hit.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np

BASE = 1_000_003
MOD = (1 << 61) - 1


def block_hash(parent: int, tokens) -> int:
    h = parent
    for t in tokens:
        h = (h * BASE + t + 1) % MOD
    return h


def chain_hashes(tokens, block_size):
    """Chain hash of every FULL block of `tokens`, in order."""
    hashes = []
    parent = 0
    n_full_blocks = len(tokens) // block_size
    for b in range(n_full_blocks):
        blk = tokens[b * block_size:(b + 1) * block_size]
        parent = block_hash(parent, blk)
        hashes.append(parent)
    return hashes


def main() -> None:
    block_size = 4

    new_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 55, 66, 77, 88, 9, 9, 9, 9, 42]

    seq_A = [1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102, 103, 9, 9, 9, 9]
    seq_B = [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511]
    seq_C = [77, 77, 77, 77, 55, 66, 77, 88, 1, 1, 1, 1]  # decoy

    cached_hashes = set()
    for seq in (seq_A, seq_B, seq_C):
        cached_hashes.update(chain_hashes(seq, block_size))

    cached_hashes_arr = np.array(sorted(cached_hashes), dtype=np.int64)
    new_tokens_arr = np.array(new_tokens, dtype=np.int64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "cached_hashes.npy", cached_hashes_arr)
    np.save(out / "new_tokens.npy", new_tokens_arr)


if __name__ == "__main__":
    main()
