def compute_flops_neutral_batch(cost_ratio, target_flops_per_token, draft_flops_per_token):
    if draft_flops_per_token <= 0:
        return 0.0
    return float(target_flops_per_token) / (float(draft_flops_per_token) * float(cost_ratio))
