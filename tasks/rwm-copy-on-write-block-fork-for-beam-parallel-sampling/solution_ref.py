def simulate_cow_blocks(block_size, initial, ops):
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
            source = op[1]
            seq = list(seqs[source])
            for bid in seq:
                refs[bid] += 1
            seqs.append(seq)
        else:
            sid, token = op[1], op[2]
            if not seqs[sid] or len(blocks[seqs[sid][-1]]) >= block_size:
                seqs[sid].append(new_block([token]))
            else:
                bid = seqs[sid][-1]
                if refs[bid] > 1:
                    refs[bid] -= 1
                    bid = new_block(blocks[bid])
                    seqs[sid][-1] = bid
                blocks[bid].append(token)

    return {"blocks": blocks, "refs": refs, "seqs": seqs}
