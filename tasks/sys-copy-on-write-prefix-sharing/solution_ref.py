def cow_kv_branches(prompt_kv, branch_kvs, block_size):
    """
    Simulate a block-paged KV cache with copy-on-write (COW) prefix sharing,
    the mechanism vLLM's PagedAttention uses to let multiple decoding
    branches share one prompt's KV blocks until they write.
    """
    P = len(prompt_kv)
    D = len(prompt_kv[0]) if P > 0 else 0
    B = len(branch_kvs)
    branch_lens = [len(bk) for bk in branch_kvs]

    full_blocks = P // block_size
    rem = P % block_size

    storage = {}     # block_id -> list of rows
    cap_used = {}     # block_id -> filled slots
    refcount = {}     # block_id -> ref count
    next_id = 0

    for i in range(full_blocks):
        storage[next_id] = [row[:] for row in prompt_kv[i * block_size:(i + 1) * block_size]]
        cap_used[next_id] = block_size
        refcount[next_id] = B
        next_id += 1

    partial_id = None
    if rem > 0:
        partial_id = next_id
        buf = [[0.0] * D for _ in range(block_size)]
        for r_idx in range(rem):
            buf[r_idx] = list(prompt_kv[full_blocks * block_size + r_idx])
        storage[partial_id] = buf
        cap_used[partial_id] = rem
        refcount[partial_id] = B
        next_id += 1

    tables = []
    for _ in range(B):
        t = list(range(full_blocks))
        if partial_id is not None:
            t.append(partial_id)
        tables.append(t)

    total_allocated = next_id

    for b in range(B):
        table = tables[b]
        toks = branch_kvs[b]
        for i in range(branch_lens[b]):
            last = table[-1] if table else None
            if last is None or cap_used[last] >= block_size:
                new_id = next_id; next_id += 1
                storage[new_id] = [[0.0] * D for _ in range(block_size)]
                cap_used[new_id] = 0
                refcount[new_id] = 1
                table.append(new_id)
                total_allocated += 1
                last = new_id
            elif refcount[last] > 1:
                new_id = next_id; next_id += 1
                storage[new_id] = [row[:] for row in storage[last]]
                cap_used[new_id] = cap_used[last]
                refcount[new_id] = 1
                refcount[last] -= 1
                table[-1] = new_id
                total_allocated += 1
                last = new_id
            storage[last][cap_used[last]] = list(toks[i])
            cap_used[last] += 1

    branch_sequences = []
    for b in range(B):
        rows = []
        for bid in tables[b]:
            rows.extend([row[:] for row in storage[bid][:cap_used[bid]]])
        branch_sequences.append(rows)

    return branch_sequences, total_allocated
