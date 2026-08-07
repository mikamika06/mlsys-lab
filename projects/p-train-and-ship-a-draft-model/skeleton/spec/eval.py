def expected_acceptance(p_target, p_draft):
    raise NotImplementedError

def average_acceptance(draft, target_probs_dict, dataset):
    raise NotImplementedError

def speculative_speedup(gamma: int, alpha: float, draft_cost_ratio: float):
    raise NotImplementedError
