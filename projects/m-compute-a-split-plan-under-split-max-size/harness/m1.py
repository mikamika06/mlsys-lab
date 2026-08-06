import ref


def check(workdir):
    from ggufsplit.planner import compute_split_plan

    out = {"plans_matched": 0.0, "configs": float(len(ref.CASES))}
    ok = 0
    for i, (tensors, max_size) in enumerate(ref.CASES):
        want = ref.compute_split_plan(tensors, max_size) if hasattr(ref, "compute_split_plan") else []
        # Fallback inline reference calculation if not present in ref
        def ref_plan(ts, ms):
            shards, cur, cur_sz = [], [], 0
            for name, sz in ts:
                if cur and (cur_sz + sz > ms):
                    shards.append(cur)
                    cur = [(name, sz)]
                    cur_sz = sz
                else:
                    cur.append((name, sz))
                    cur_sz += sz
            if cur:
                shards.append(cur)
            return shards

        expected = ref_plan(tensors, max_size)
        got = compute_split_plan(tensors, max_size)
        if got == expected:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, reference {expected}"
    out["plans_matched"] = float(ok)
    return out
