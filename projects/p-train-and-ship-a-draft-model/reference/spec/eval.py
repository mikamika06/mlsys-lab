import numpy as np

def expected_acceptance(p_target, p_draft):
    return float(np.sum(np.minimum(p_target, p_draft)))

def average_acceptance(draft, target_probs_dict, dataset):
    accs = []
    for token in dataset:
        p_t = target_probs_dict[token]
        p_d = draft.get_probs(token)
        accs.append(expected_acceptance(p_t, p_d))
    return float(np.mean(accs))

def speculative_speedup(gamma: int, alpha: float, draft_cost_ratio: float):
    expected_tokens = (1.0 - alpha**(gamma + 1.0)) / (1.0 - alpha)
    step_cost = 1.0 + gamma * draft_cost_ratio
    return expected_tokens / step_cost
