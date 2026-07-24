def _oracle(ops, block_size):
    blocks = {}
    owners = {}
    sequences = {}
    next_block = 0

    def alloc(tokens):
        nonlocal next_block
        b = next_block
        next_block += 1
        blocks[b] = list(tokens)
        owners[b] = 1
        return b

    def retain(b):
        owners[b] += 1

    def release(b):
        owners[b] -= 1
        if owners[b] == 0:
            del owners[b]
            del blocks[b]

    def read_seq(seq):
        out = []
        for b in sequences[seq]:
            out.extend(blocks[b])
        return out

    for op in ops:
        kind = op[0]
        if kind == "create":
            _, sid, tokens = op
            seq_blocks = []
            for i in range(0, len(tokens), block_size):
                seq_blocks.append(alloc(tokens[i:i + block_size]))
            sequences[sid] = seq_blocks
        elif kind == "fork":
            _, new_sid, parent = op
            sequences[new_sid] = list(sequences[parent])
            for b in sequences[new_sid]:
                retain(b)
        elif kind == "append":
            _, sid, token = op
            current = sequences[sid]
            if not current or len(blocks[current[-1]]) >= block_size:
                current.append(alloc([token]))
            else:
                old = current[-1]
                if owners[old] > 1:
                    release(old)
                    current[-1] = alloc(blocks[old] + [token])
                else:
                    blocks[old].append(token)
        elif kind == "delete":
            _, sid = op
            for b in sequences[sid]:
                release(b)
            del sequences[sid]

    return {k: read_seq(k) for k in sorted(sequences)}


def grade(sol, fx) -> dict:
    cases = [
        [
            ("create", "a", [1, 2, 3, 4]),
            ("fork", "b", "a"),
            ("append", "a", 5),
            ("delete", "a"),
        ],
        [
            ("create", "x", [7, 8, 9, 10, 11]),
            ("fork", "y", "x"),
            ("fork", "z", "y"),
            ("append", "y", 12),
            ("delete", "z"),
            ("append", "x", 13),
        ],
        [
            ("create", "p", [1, 2, 3]),
            ("fork", "q", "p"),
            ("append", "q", 4),
            ("append", "p", 5),
            ("delete", "q"),
        ],
    ]

    ok = 1.0
    for ops in cases:
        try:
            got = sol.replay_kv_trace(ops, 4)
            ref = _oracle(ops, 4)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
