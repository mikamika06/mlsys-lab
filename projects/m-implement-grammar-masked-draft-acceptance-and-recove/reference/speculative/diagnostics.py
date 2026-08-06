def diagnose_collapse(run_a_metrics, run_b_metrics):
    loss_a = run_a_metrics.get("acceptance_loss", 0.0)
    loss_b = run_b_metrics.get("acceptance_loss", 0.0)
    if loss_b > loss_a + 0.2:
        return "collapsed"
    return "stable"
