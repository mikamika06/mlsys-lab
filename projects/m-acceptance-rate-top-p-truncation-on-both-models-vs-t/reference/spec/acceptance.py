import numpy as np
from spec.sampling import apply_temperature_and_topp, compute_acceptance_prob


def evaluate_acceptance_rate(target_logits_seq: np.ndarray, draft_logits_seq: np.ndarray, draft_tokens: np.ndarray, temperature: float, top_p_target: float, top_p_draft: float) -> dict:
    seq_len = len(draft_tokens)
    accepted_flags = []
    theoretical_probs = []
    for i in range(seq_len):
        t_id = int(draft_tokens[i])
        p_l = target_logits_seq[i]
        q_l = draft_logits_seq[i]
        prob = compute_acceptance_prob(p_l, q_l, t_id, temperature, top_p_target, top_p_draft)
        theoretical_probs.append(prob)
        p_dist = apply_temperature_and_topp(p_l, temperature, top_p_target)
        q_dist = apply_temperature_and_topp(q_l, temperature, top_p_draft)
        expected_acc = np.sum(np.minimum(p_dist, q_dist))
        accepted_flags.append(expected_acc)
    empirical_mean = float(np.mean(theoretical_probs))
    expected_mean = float(np.mean(accepted_flags))
    return {
        "empirical_acceptance_rate": empirical_mean,
        "expected_acceptance_rate": expected_mean,
        "per_token_prob": theoretical_probs
    }
