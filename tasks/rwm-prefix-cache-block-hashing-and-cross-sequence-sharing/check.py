import numpy as np


def _block_hashes(seq: list[int], block_size: int) -> list[int]:
    n_full = len(seq) // block_size
    parent = None
    hashes = []
    for i in range(n_full):
        block = tuple(seq[i * block_size:(i + 1) * block_size])
        h = hash((parent, block))
        hashes.append(h)
        parent = h
    return hashes


def _oracle(sequences: list[list[int]], block_size: int) -> dict:
    block_hashes = [_block_hashes(seq, block_size) for seq in sequences]
    unique = set()
    for hs in block_hashes:
        unique.update(hs)
    total_naive = sum(len(hs) for hs in block_hashes)
    num_physical_blocks = len(unique)
    blocks_saved = total_naive - num_physical_blocks
    return {
        "block_hashes": block_hashes,
        "num_physical_blocks": num_physical_blocks,
        "blocks_saved": blocks_saved,
    }


def _gen_sequences() -> list[list[int]]:
    rng = np.random.default_rng(0)
    vocab = 50
    base = rng.integers(0, vocab, size=12).tolist()  # 3 full blocks (block_size=4)
    seq_a = base + rng.integers(0, vocab, size=8).tolist()          # shares 3 blocks with b, c
    seq_b = base + rng.integers(0, vocab, size=8).tolist()          # shares 3 blocks with a
    seq_c = base[:8] + rng.integers(0, vocab, size=12).tolist()     # shares 2 blocks with a, b
    seq_d = rng.integers(0, vocab, size=20).tolist()                # unrelated
    seq_e = list(seq_a)                                             # exact duplicate of a
    return [seq_a, seq_b, seq_c, seq_d, seq_e]


def grade(sol, fx) -> dict:
    block_size = 4
    sequences = _gen_sequences()
    ref = _oracle(sequences, block_size)

    try:
        got = sol.prefix_block_share([list(s) for s in sequences], block_size)
        got_hashes = got["block_hashes"]
        got_num = int(got["num_physical_blocks"])
        got_saved = int(got["blocks_saved"])
    except Exception:
        return {"exact_match": 0.0}

    if not isinstance(got_hashes, list) or len(got_hashes) != len(ref["block_hashes"]):
        return {"exact_match": 0.0}

    for got_hs, ref_hs in zip(got_hashes, ref["block_hashes"]):
        if list(got_hs) != list(ref_hs):
            return {"exact_match": 0.0}

    if got_num != ref["num_physical_blocks"] or got_saved != ref["blocks_saved"]:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0}
