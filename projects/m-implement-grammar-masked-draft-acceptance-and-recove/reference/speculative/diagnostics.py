def diagnose_collapse(metrics_a, metrics_b):
    loss_a = metrics_a.get("loss", 0.0)
    loss_b = metrics_b.get("loss", 0.0)
    if loss_a > loss_b + 0.2:
        return "run_a"
    if loss_b > loss_a + 0.2:
        return "run_b"
    return "neither"
