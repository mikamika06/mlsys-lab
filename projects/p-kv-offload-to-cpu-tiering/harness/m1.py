def check(workdir):
    from kvtier.model import CostModel
    m = {"cost_model_ok": 0.0}
    cm = CostModel(10.0)
    cost = cm.transfer_cost(1000)
    if cost > 0:
        m["cost_model_ok"] = 1.0
    return m
