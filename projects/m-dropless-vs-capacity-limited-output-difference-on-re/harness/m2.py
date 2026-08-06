import ref

def check(workdir):
    from moe_routing.capacity import select_capacity_factor
    tokens, logits, top_k = ref.generate_test_data(seed=456)
    max_drop = 0.1
    want = ref.find_min_capacity_factor(tokens, logits, top_k, max_drop)
    got = select_capacity_factor(tokens, logits, top_k, max_drop)

    match = 1.0 if abs(want - got) < 1e-4 else 0.0
    return {"capacity_matched": match}
