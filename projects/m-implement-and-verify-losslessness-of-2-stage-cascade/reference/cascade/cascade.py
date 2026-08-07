import numpy as np


def sample_2stage_cascade(draft1_probs, draft2_probs, target_probs, seed=42):
    rng = np.random.default_rng(seed)
    gamma1 = draft1_probs.shape[0]
    gamma2 = draft2_probs.shape[0]

    tokens = []

    # Stage 1: Generate gamma1 tokens from draft1
    d1_tokens = []
    for i in range(gamma1):
        tok = rng.choice(len(draft1_probs[i]), p=draft1_probs[i])
        d1_tokens.append(tok)

    # Stage 2: Filter with draft2
    d2_accepted = []
    for i in range(gamma1):
        tok = d1_tokens[i]
        p1 = draft1_probs[i, tok]
        p2 = draft2_probs[i, tok]
        u = rng.uniform()
        if u < min(1.0, p2 / p1):
            d2_accepted.append(tok)
            if len(d2_accepted) == gamma2:
                break
        else:
            resample_p = np.maximum(0.0, draft2_probs[i] - draft1_probs[i])
            s = resample_p.sum()
            if s > 0:
                resample_p = resample_p / s
            else:
                resample_p = draft2_probs[i]
            resample_tok = rng.choice(len(resample_p), p=resample_p)
            d2_accepted.append(resample_tok)
            break

    # Stage 3: Validate against target
    for i, tok in enumerate(d2_accepted):
        p2 = draft2_probs[i, tok]
        p_target = target_probs[i, tok]
        u = rng.uniform()
        if u < min(1.0, p_target / p2):
            tokens.append(tok)
        else:
            resample_p = np.maximum(0.0, target_probs[i] - draft2_probs[i])
            s = resample_p.sum()
            if s > 0:
                resample_p = resample_p / s
            else:
                resample_p = target_probs[i]
            resample_tok = rng.choice(len(resample_p), p=resample_p)
            tokens.append(resample_tok)
            return tokens

    # Extra target token if all draft tokens accepted
    n_acc = len(tokens)
    extra_tok = rng.choice(len(target_probs[n_acc]), p=target_probs[n_acc])
    tokens.append(extra_tok)
    return tokens
