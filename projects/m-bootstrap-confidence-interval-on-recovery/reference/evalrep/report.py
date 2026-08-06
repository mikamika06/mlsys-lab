from evalrep.bootstrap import bootstrap_recovery_ci

def format_recovery_report(base_scores, quant_scores):
    ci = bootstrap_recovery_ci(base_scores, quant_scores)
    return f"Recovery: {ci['mean']:.2f}% (95% CI [{ci['lower']:.2f}, {ci['upper']:.2f}])"
