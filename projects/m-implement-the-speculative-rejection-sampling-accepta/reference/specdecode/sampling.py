import numpy as np


def sample_speculative(target_probs, draft_probs, draft_tokens, u_accept, u_resample):
    """Speculative rejection sampling algorithm for a batch of draft tokens."""
    gamma = len(draft_tokens)
    accepted = []
    for i in range(gamma):
        tok = int(draft_tokens[i])
        p = float(target_probs[i, tok])
        q = float(draft_probs[i, tok])
        acc_prob = min(1.0, p / q) if q > 0.0 else 0.0
        if u_accept[i] < acc_prob:
            accepted.append(tok)
        else:
            diff = np.maximum(0.0, target_probs[i] - draft_probs[i])
            total = float(np.sum(diff))
            if total > 0.0:
                norm_diff = diff / total
            else:
                norm_diff = target_probs[i]
            cdf = np.cumsum(norm_diff)
            resampled_tok = int(np.searchsorted(cdf, u_resample[i]))
            resampled_tok = min(resampled_tok, len(norm_diff) - 1)
            accepted.append(resampled_tok)
            return np.array(accepted, dtype=int)

    cdf = np.cumsum(target_probs[gamma])
    bonus_tok = int(np.searchsorted(cdf, u_resample[gamma]))
    bonus_tok = min(bonus_tok, len(target_probs[gamma]) - 1)
    accepted.append(bonus_tok)
    return np.array(accepted, dtype=int)
