import numpy as np


def rejection_sample(target_probs, draft_probs, draft_tokens, rng):
    """Speculative rejection sampling over candidate draft tokens."""
    k = len(draft_tokens)
    accepted_tokens = []
    num_accepted = 0

    for i in range(k):
        t = int(draft_tokens[i])
        p = float(target_probs[i, t])
        q = float(draft_probs[i, t])

        ratio = p / q if q > 0 else (1.0 if p > 0 else 0.0)
        u = float(rng.uniform(0.0, 1.0))

        if u <= min(1.0, ratio):
            accepted_tokens.append(t)
            num_accepted += 1
        else:
            p_prime = np.maximum(0.0, target_probs[i] - draft_probs[i])
            total = float(np.sum(p_prime))
            if total > 0:
                p_prime = p_prime / total
            else:
                p_prime = target_probs[i] / np.sum(target_probs[i])

            resampled = int(rng.choice(len(p_prime), p=p_prime))
            accepted_tokens.append(resampled)
            break

    if num_accepted == k:
        extra = int(rng.choice(target_probs.shape[1], p=target_probs[k]))
        accepted_tokens.append(extra)

    return np.array(accepted_tokens, dtype=np.int64), num_accepted
