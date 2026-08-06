def diagnose_run(acceptance_rate, speedup_ratio, vocab_overlap_ratio):
    if vocab_overlap_ratio < 0.75:
        return "mismatched_family_fatal"
    if speedup_ratio < 1.0:
        return "net_harming"
    if acceptance_rate < 0.5:
        return "suboptimal_acceptance"
    return "net_helping"
