import numpy as np


def verify_and_recover_draft(draft_tokens, draft_probs, target_logits, fsm, start_state=0):
    accepted = []
    curr_state = start_state
    k = len(draft_tokens)

    for i in range(k):
        tok = draft_tokens[i]
        q_i = draft_probs[i]

        raw_logits = target_logits[i]
        masked_logits = fsm.mask_logits(curr_state, raw_logits)
        exp_l = np.exp(masked_logits - np.max(masked_logits))
        p_i = exp_l / np.sum(exp_l)

        allowed_tokens = fsm.transitions.get(curr_state, {})
        if tok not in allowed_tokens:
            p_target = 0.0
        else:
            p_target = p_i[tok]

        p_draft = q_i[tok]
        ratio = p_target / p_draft if p_draft > 0 else (1.0 if p_target > 0 else 0.0)
        accept_prob = min(1.0, ratio)

        if accept_prob >= 1.0:
            accepted.append(tok)
            curr_state = fsm.step(curr_state, tok)
        else:
            diff = np.maximum(0.0, p_i - q_i)
            sum_diff = np.sum(diff)
            if sum_diff > 1e-12:
                norm_p = diff / sum_diff
            else:
                norm_p = p_i
            resampled_tok = int(np.argmax(norm_p))
            accepted.append(resampled_tok)
            curr_state = fsm.step(curr_state, resampled_tok)
            return accepted, curr_state, False

    raw_logits_bonus = target_logits[k]
    masked_logits_bonus = fsm.mask_logits(curr_state, raw_logits_bonus)
    exp_l_bonus = np.exp(masked_logits_bonus - np.max(masked_logits_bonus))
    p_bonus = exp_l_bonus / np.sum(exp_l_bonus)
    bonus_tok = int(np.argmax(p_bonus))
    accepted.append(bonus_tok)
    curr_state = fsm.step(curr_state, bonus_tok)

    return accepted, curr_state, True
