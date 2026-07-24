def _oracle(block_size, initial, ops):
    blocks = []
    refs = []
    seqs = []

    def new_block(values):
        blocks.append(list(values))
        refs.append(1)
        return len(blocks) - 1

    first = []
    for i in range(0, len(initial), block_size):
        first.append(new_block(initial[i:i + block_size]))
    seqs.append(first)

    for op in ops:
        if op[0] == "fork":
            src = op[1]
            copied = list(seqs[src])
            for bid in copied:
                refs[bid] += 1
            seqs.append(copied)
        else:
            sid, token = op[1], op[2]
            if not seqs[sid] or len(blocks[seqs[sid][-1]]) >= block_size:
                seqs[sid].append(new_block([token]))
            else:
                old = seqs[sid][-1]
                if refs[old] > 1:
                    refs[old] -= 1
                    copied = new_block(blocks[old])
                    seqs[sid][-1] = copied
                blocks[seqs[sid][-1]].append(token)

    return {"blocks": blocks, "refs": refs, "seqs": seqs}


def grade(sol, fx) -> dict:
    cases = [
        (
            4,
            [1, 2, 3, 4],
            [
                ("fork", 0),
                ("append", 1, 5),
                ("append", 0, 6),
            ],
        ),
        (
            3,
            [10, 11, 12, 13, 14],
            [
                ("fork", 0),
                ("fork", 1),
                ("append", 2, 15),
                ("append", 0, 16),
                ("append", 1, 17),
            ],
        ),
        (
            2,
            [0, 1],
            [
                ("fork", 0),
                ("append", 1, 2),
                ("fork", 0),
                ("append", 2, 3),
                ("append", 0, 4),
            ],
        ),
    ]

    ok = 1.0
    for block_size, initial, ops in cases:
        expected = _oracle(block_size, initial, ops)
        try:
            got = sol.simulate_cow_blocks(block_size, list(initial), list(ops))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
