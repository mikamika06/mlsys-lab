import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from kvplan.planner import calculate_paged_kv_plan, simulate_block_allocation
    except Exception as e:
        return {"plans_matched": 0.0, "_note": f"Import failed: {e}"}

    matched = 0
    for cfg in ref.TEST_CONFIGS:
        want = ref.calculate_paged_kv_plan(cfg["seq_lens"], cfg["block_size"], cfg["page_budget"])
        got = calculate_paged_kv_plan(cfg["seq_lens"], cfg["block_size"], cfg["page_budget"])
        if got == want:
            matched += 1

    pattern = [
        {"action": "arrive", "seq_len": 32},
        {"action": "arrive", "seq_len": 64},
        {"action": "depart", "seq_len": 32},
        {"action": "arrive", "seq_len": 128}
    ]
    want_sim = ref.simulate_block_allocation(pattern, 16, 10)
    got_sim = simulate_block_allocation(pattern, 16, 10)
    if got_sim == want_sim:
        matched += 1

    return {"plans_matched": float(matched)}
