def replay_kv_trace(trace, block_size, num_blocks):
    blocks = [None] * num_blocks
    refs = [0] * num_blocks
    seqs = {}

    def new_block(data):
        for i in range(num_blocks):
            if refs[i] == 0:
                blocks[i] = list(data)
                refs[i] = 1
                return i
        raise RuntimeError("out of blocks")

    def unique(sid, idx):
        bid = seqs[sid]["blocks"][idx]
        if refs[bid] > 1:
            refs[bid] -= 1
            bid2 = new_block(blocks[bid])
            seqs[sid]["blocks"][idx] = bid2
            return bid2
        return bid

    for op in trace:
        if op[0] == "alloc":
            _, sid, tokens = op
            seqs[sid] = {"blocks": [], "tokens": list(tokens)}
            for i in range(0, len(tokens), block_size):
                seqs[sid]["blocks"].append(new_block(tokens[i:i + block_size]))
        elif op[0] == "branch":
            _, sid, src = op
            seqs[sid] = {"blocks": list(seqs[src]["blocks"]), "tokens": list(seqs[src]["tokens"])}
            for b in seqs[sid]["blocks"]:
                refs[b] += 1
        elif op[0] == "append":
            _, sid, token = op
            seqs[sid]["tokens"].append(token)
            idx = (len(seqs[sid]["tokens"]) - 1) // block_size
            if idx == len(seqs[sid]["blocks"]):
                seqs[sid]["blocks"].append(new_block([]))
            b = unique(sid, idx)
            off = (len(seqs[sid]["tokens"]) - 1) % block_size
            while len(blocks[b]) <= off:
                blocks[b].append(None)
            blocks[b][off] = token
        elif op[0] == "free":
            _, sid = op
            for b in seqs[sid]["blocks"]:
                refs[b] -= 1
            del seqs[sid]

    table = sorted((sid, list(v["blocks"])) for sid, v in seqs.items())
    bt = [(i, list(b)) for i, b in enumerate(blocks) if refs[i]]
    rc = [(i, r) for i, r in enumerate(refs) if r]
    return table, bt, rc
