def replay_kv_trace(ops, block_size):
    blocks = {}
    refs = {}
    sequences = {}
    next_block = 0

    def new_block(tokens):
        nonlocal next_block
        b = next_block
        next_block += 1
        blocks[b] = list(tokens)
        refs[b] = 1
        return b

    def retain(b):
        refs[b] += 1

    def release(b):
        refs[b] -= 1
        if refs[b] == 0:
            del refs[b]
            del blocks[b]

    def logical(sid):
        result = []
        for b in sequences[sid]:
            result.extend(blocks[b])
        return result

    for op in ops:
        kind = op[0]

        if kind == "create":
            _, sid, tokens = op
            sequences[sid] = []
            for i in range(0, len(tokens), block_size):
                sequences[sid].append(new_block(tokens[i:i + block_size]))

        elif kind == "fork":
            _, sid, parent = op
            sequences[sid] = list(sequences[parent])
            for b in sequences[sid]:
                retain(b)

        elif kind == "append":
            _, sid, token = op
            seq = sequences[sid]
            if not seq or len(blocks[seq[-1]]) == block_size:
                seq.append(new_block([token]))
            else:
                b = seq[-1]
                if refs[b] > 1:
                    old = list(blocks[b])
                    release(b)
                    seq[-1] = new_block(old + [token])
                else:
                    blocks[b].append(token)

        elif kind == "delete":
            _, sid = op
            for b in sequences[sid]:
                release(b)
            del sequences[sid]

    return {sid: logical(sid) for sid in sorted(sequences)}
