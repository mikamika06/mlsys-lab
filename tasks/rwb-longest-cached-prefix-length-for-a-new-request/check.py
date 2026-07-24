import numpy as np

BASE = 1_000_003
MOD = (1 << 61) - 1


def _block_hash(parent, tokens):
    h = parent
    for t in tokens:
        h = (h * BASE + t + 1) % MOD
    return h


def _chain_hashes(tokens, block_size):
    hashes = []
    parent = 0
    n_full = len(tokens) // block_size
    for b in range(n_full):
        blk = tokens[b * block_size:(b + 1) * block_size]
        parent = _block_hash(parent, blk)
        hashes.append(parent)
    return hashes


def _oracle_hits(cached_hashes, new_tokens, block_size):
    n_full = len(new_tokens) // block_size
    parent = 0
    hits = 0
    for b in range(n_full):
        blk = new_tokens[b * block_size:(b + 1) * block_size]
        h = _block_hash(parent, blk)
        if h not in cached_hashes:
            break
        hits += 1
        parent = h
    return hits


def _synthetic_cases():
    rng = np.random.default_rng(61)
    cases = []
    for _ in range(6):
        block_size = int(rng.integers(2, 6))
        vocab = 40
        shared_blocks = int(rng.integers(0, 4))
        total_blocks = shared_blocks + int(rng.integers(1, 4))

        base_seq = rng.integers(0, vocab, size=total_blocks * block_size).tolist()
        new_tokens = list(base_seq[:shared_blocks * block_size])
        # diverge after the shared prefix, then add a few more (possibly
        # partial) tokens
        tail_len = int(rng.integers(1, 2 * block_size + 1))
        new_tokens += rng.integers(vocab, vocab + 50, size=tail_len).tolist()

        cached = set(_chain_hashes(base_seq, block_size))
        # a bit of unrelated noise in the cache
        noise = rng.integers(0, vocab, size=3 * block_size).tolist()
        cached.update(_chain_hashes(noise, block_size))

        cases.append((cached, new_tokens, block_size))
    return cases


def grade(sol, fx) -> dict:
    fixture_cached = set(int(x) for x in fx["cached_hashes"].tolist())
    fixture_new = fx["new_tokens"].tolist()
    cases = [(fixture_cached, fixture_new, 4)] + _synthetic_cases()

    total = 0
    correct = 0
    for cached_hashes, new_tokens, block_size in cases:
        ref = _oracle_hits(cached_hashes, new_tokens, block_size)
        total += 1
        try:
            got = sol.longest_cached_prefix_blocks(set(cached_hashes), list(new_tokens), block_size)
            if int(got) == ref:
                correct += 1
        except Exception:
            pass

    return {"exact_match": (correct / total) if total else 0.0}
