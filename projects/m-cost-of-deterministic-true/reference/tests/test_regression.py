from detcost.tradeoff import optimize_checkpointing

def test_checkpointing_budget_and_validity():
    def dummy_cost(layer_idx, is_ckpt):
        time = 1.5 if is_ckpt else 1.0
        mem = 100 if is_ckpt else 500
        return time, mem

    layers = 4
    budget = 1200
    plan = optimize_checkpointing(layers, budget, dummy_cost)
    assert len(plan) == layers

    total_mem = 0
    current_seg = 0
    for i in range(layers):
        _, m = dummy_cost(i, plan[i])
        if plan[i]:
            current_seg = 0
        else:
            current_seg += 1
        total_mem = max(total_mem, m + current_seg * 1024)
    assert total_mem <= budget
