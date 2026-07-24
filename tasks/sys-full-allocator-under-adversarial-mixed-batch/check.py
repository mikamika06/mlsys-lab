def _oracle(trace, block_size, num_blocks):
    blocks = [None] * num_blocks
    refs = [0] * num_blocks
    seqs = {}

    def alloc_block(tokens):
        for i in range(num_blocks):
            if refs[i] == 0:
                blocks[i] = list(tokens)
                refs[i] = 1
                return i
        raise RuntimeError("out of blocks")

    def ensure_unique(seq_id, logical_idx):
        bid = seqs[seq_id]["blocks"][logical_idx]
        if refs[bid] > 1:
            refs[bid] -= 1
            new = alloc_block(blocks[bid])
            seqs[seq_id]["blocks"][logical_idx] = new
            return new
        return bid

    for op in trace:
        if op[0] == "alloc":
            _, sid, tokens = op
            table = []
            for i in range(0, len(tokens), block_size):
                table.append(alloc_block(tokens[i:i + block_size]))
            seqs[sid] = {"blocks": table, "tokens": list(tokens)}
        elif op[0] == "branch":
            _, new, src = op
            seqs[new] = {
                "blocks": list(seqs[src]["blocks"]),
                "tokens": list(seqs[src]["tokens"]),
            }
            for bid in seqs[new]["blocks"]:
                refs[bid] += 1
        elif op[0] == "append":
            _, sid, token = op
            seqs[sid]["tokens"].append(token)
            logical = (len(seqs[sid]["tokens"]) - 1) // block_size
            if logical == len(seqs[sid]["blocks"]):
                seqs[sid]["blocks"].append(alloc_block([]))
            bid = ensure_unique(sid, logical)
            offset = (len(seqs[sid]["tokens"]) - 1) % block_size
            while len(blocks[bid]) <= offset:
                blocks[bid].append(None)
            blocks[bid][offset] = token
        elif op[0] == "free":
            _, sid = op
            for bid in seqs[sid]["blocks"]:
                refs[bid] -= 1
            del seqs[sid]

    table = sorted((sid, list(v["blocks"])) for sid, v in seqs.items())
    bt = []
    rc = []
    for i, r in enumerate(refs):
        if r:
            bt.append((i, list(blocks[i])))
            rc.append((i, r))
    return table, bt, rc


def grade(sol, fx) -> dict:
    traces = [
        (
            [
                ("alloc", 1, [1, 2, 3]),
                ("branch", 2, 1),
                ("append", 2, 4),
                ("free", 1),
            ],
            2,
            5,
        ),
        (
            [
                ("alloc", 3, [9, 8, 7, 6]),
                ("branch", 4, 3),
                ("branch", 5, 4),
                ("append", 5, 5),
                ("append", 4, 10),
                ("free", 3),
                ("free", 4),
            ],
            2,
            8,
        ),
        (
            [
                ("alloc", 1, [0, 1, 2, 3, 4]),
                ("append", 1, 5),
                ("alloc", 2, [7]),
                ("branch", 3, 1),
                ("free", 2),
                ("append", 3, 8),
            ],
            3,
            7,
        ),
    ]

    ok = 1.0
    for trace, block_size, num_blocks in traces:
        try:
            got = sol.replay_kv_trace(trace, block_size, num_blocks)
            ref = _oracle(trace, block_size, num_blocks)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
