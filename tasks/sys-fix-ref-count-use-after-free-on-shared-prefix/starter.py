def replay_kv_trace(ops, block_size):
    blocks = {}
    sequences = {}
    next_block = 0

    def alloc(tokens):
        nonlocal next_block
        b = next_block
        next_block += 1
        blocks[b] = list(tokens)
        return b

    def read(sid):
        out = []
        for b in sequences[sid]:
            out.extend(blocks[b])
        return out

    for op in ops:
        kind = op[0]

        if kind == "create":
            _, sid, tokens = op
            sequences[sid] = []
            for i in range(0, len(tokens), block_size):
                sequences[sid].append(alloc(tokens[i:i + block_size]))

        elif kind == "fork":
            _, sid, parent = op
            sequences[sid] = list(sequences[parent])

        elif kind == "append":
            _, sid, token = op
            seq = sequences[sid]
            if not seq or len(blocks[seq[-1]]) == block_size:
                seq.append(alloc([token]))
            else:
                # TODO: this mutates a shared block and causes use-after-free
                # style corruption because forked sequences share this block.
                blocks[seq[-1]].append(token)

        elif kind == "delete":
            _, sid = op
            del sequences[sid]

    return {sid: read(sid) for sid in sorted(sequences)}
