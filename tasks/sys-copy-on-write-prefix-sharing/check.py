import numpy as np


def _simulate_blocks(P, block_size, branch_lens):
    """Reference COW block-manager simulation. Returns total physical
    blocks ever allocated. Branches processed 0..B-1, tokens in order."""
    full_blocks = P // block_size
    rem = P % block_size
    next_id = 0
    cap_used = {}
    refcount = {}
    B = len(branch_lens)
    for _ in range(full_blocks):
        cap_used[next_id] = block_size
        refcount[next_id] = B
        next_id += 1
    partial_id = None
    if rem > 0:
        partial_id = next_id
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

    for b, L in enumerate(branch_lens):
        table = tables[b]
        for _ in range(L):
            last = table[-1] if table else None
            if last is None or cap_used[last] >= block_size:
                new_id = next_id; next_id += 1
                cap_used[new_id] = 0
                refcount[new_id] = 1
                table.append(new_id)
                total_allocated += 1
                last = new_id
            elif refcount[last] > 1:
                new_id = next_id; next_id += 1
                cap_used[new_id] = cap_used[last]
                refcount[new_id] = 1
                refcount[last] -= 1
                table[-1] = new_id
                total_allocated += 1
                last = new_id
            cap_used[last] += 1
    return total_allocated


def _gen_case(rng):
    D = 3
    P = int(rng.integers(3, 13))
    block_size = int(rng.integers(2, 6))
    B = int(rng.integers(2, 5))
    branch_lens = [int(rng.integers(0, 7)) for _ in range(B)]
    prompt_kv = rng.standard_normal((P, D))
    branch_kvs = [rng.standard_normal((L, D)) for L in branch_lens]
    return prompt_kv, branch_kvs, block_size, branch_lens


def grade(sol, fx) -> dict:
    cases = []
    # fixed example from task.md: P=5, block_size=4, two branches of len 2
    rng0 = np.random.default_rng(1)
    prompt0 = rng0.standard_normal((5, 3))
    b0 = [rng0.standard_normal((2, 3)), rng0.standard_normal((2, 3))]
    cases.append((prompt0, b0, 4, [2, 2]))

    rng = np.random.default_rng(0)
    for _ in range(8):
        cases.append(_gen_case(rng))

    worst_err = 0.0
    block_ok = 1.0
    for prompt_kv, branch_kvs, block_size, branch_lens in cases:
        expected_seqs = [np.concatenate([prompt_kv, bk], axis=0) for bk in branch_kvs]
        expected_blocks = _simulate_blocks(prompt_kv.shape[0], block_size, branch_lens)
        try:
            got_seqs, got_blocks = sol.cow_kv_branches(
                prompt_kv.copy(), [bk.copy() for bk in branch_kvs], block_size
            )
            if len(got_seqs) != len(expected_seqs):
                worst_err = float("inf")
                block_ok = 0.0
                break
            for g, e in zip(got_seqs, expected_seqs):
                g = np.asarray(g, dtype=np.float64)
                if g.shape != e.shape:
                    worst_err = float("inf")
                    break
                worst_err = max(worst_err, float(np.max(np.abs(g - e))) if e.size else 0.0)
            if int(got_blocks) != int(expected_blocks):
                block_ok = 0.0
        except Exception:
            worst_err = float("inf")
            block_ok = 0.0
            break

    return {"max_abs_err": worst_err, "block_count_exact_match": block_ok}
