import numpy as np


def cow_kv_branches(prompt_kv, branch_kvs, block_size):
    """
    Simulate a block-paged KV cache with copy-on-write (COW) prefix sharing,
    the mechanism vLLM's PagedAttention uses to let multiple decoding
    branches share one prompt's KV blocks until they write.
    """
    P, D = prompt_kv.shape
    B = len(branch_kvs)
    branch_lens = [bk.shape[0] for bk in branch_kvs]

    full_blocks = P // block_size
    rem = P % block_size

    storage = {}     # block_id -> (block_size, D) array
    cap_used = {}     # block_id -> filled slots
    refcount = {}     # block_id -> ref count
    next_id = 0

    for i in range(full_blocks):
        storage[next_id] = prompt_kv[i * block_size:(i + 1) * block_size].copy()
        cap_used[next_id] = block_size
        refcount[next_id] = B
        next_id += 1

    partial_id = None
    if rem > 0:
        partial_id = next_id
        buf = np.zeros((block_size, D), dtype=prompt_kv.dtype)
        buf[:rem] = prompt_kv[full_blocks * block_size:]
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
                storage[new_id] = np.zeros((block_size, D), dtype=prompt_kv.dtype)
                cap_used[new_id] = 0
                refcount[new_id] = 1
                table.append(new_id)
                total_allocated += 1
                last = new_id
            elif refcount[last] > 1:
                new_id = next_id; next_id += 1
                storage[new_id] = storage[last].copy()
                cap_used[new_id] = cap_used[last]
                refcount[new_id] = 1
                refcount[last] -= 1
                table[-1] = new_id
                total_allocated += 1
                last = new_id
            storage[last][cap_used[last]] = toks[i]
            cap_used[last] += 1

    branch_sequences = []
    for b in range(B):
        rows = []
        for bid in tables[b]:
            rows.append(storage[bid][:cap_used[bid]])
        branch_sequences.append(np.concatenate(rows, axis=0) if rows else np.zeros((0, D)))

    return branch_sequences, total_allocated
