def check(workdir):
    from moe.cost import estimate_transfer_cost

    m = {"cost_accuracy": 0.0}
    cost = estimate_transfer_cost(1000, 500)
    if abs(cost - 2.0) < 1e-5:
        m["cost_accuracy"] = 1.0
    return m
