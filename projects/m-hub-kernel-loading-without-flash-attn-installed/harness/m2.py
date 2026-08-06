import ref


def check(workdir):
    from hf_attn.cost import estimate_train_step_cost, compare_backend_costs

    out = {"cost_matched": 0.0}
    ok = 0
    total = len(ref.COST_CASES)

    for i, case in enumerate(ref.COST_CASES):
        bs = case["batch_size"]
        sl = case["seq_len"]
        nh = case["num_heads"]
        hd = case["head_dim"]
        backends = case["backends"]

        want_cmp = ref.compare_backend_costs(bs, sl, nh, hd, backends)
        try:
            got_cmp = compare_backend_costs(bs, sl, nh, hd, backends)
            if got_cmp == want_cmp:
                ok += 1
            else:
                out["_note"] = f"case {i}: expected {want_cmp}, got {got_cmp}"
                break
        except Exception as e:
            out["_note"] = f"case {i} raised exception: {e}"
            break

    if ok == total:
        out["cost_matched"] = 1.0
    return out
