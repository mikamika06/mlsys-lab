import ref

def check(workdir):
    try:
        from caching.blocks import surviving_blocks
    except ImportError:
        return {"exact_match": 0.0, "_note": "Could not import surviving_blocks"}

    import random
    rng = random.Random(43)

    ok = 0
    total = 100
    for _ in range(total):
        block_contents = {}
        cached_seqs = []

        for b in range(1, 20):
            block_contents[b] = [rng.randint(1, 5) for _ in range(4)]

        for _ in range(3):
            seq = [rng.randint(1, 19) for _ in range(rng.randint(2, 6))]
            cached_seqs.append(seq)

        target_seq = rng.choice(cached_seqs)
        cut_blocks = rng.randint(0, len(target_seq))

        new_prompt = []
        for b in target_seq[:cut_blocks]:
            new_prompt.extend(block_contents[b])

        if cut_blocks < len(target_seq) and rng.choice([True, False]):
            partial_block = target_seq[cut_blocks]
            ptokens = block_contents[partial_block][:rng.randint(1, 3)]
            new_prompt.extend(ptokens)

        new_prompt.extend([rng.randint(6, 10) for _ in range(rng.randint(0, 5))])

        want = ref.surviving_blocks(new_prompt, cached_seqs, block_contents)
        try:
            got = surviving_blocks(new_prompt, cached_seqs, block_contents)
            if got == want:
                ok += 1
        except Exception:
            pass

    return {"exact_match": ok / total}
