import numpy as np


def simulate_acceptance_rates(draft_fp16_logits, draft_int8_logits, target_logits, gamma=4):
    fp16_logits = np.asarray(draft_fp16_logits, dtype=np.float32)
    int8_logits = np.asarray(draft_int8_logits, dtype=np.float32)
    tgt_logits = np.asarray(target_logits, dtype=np.float32)

    n_runs = fp16_logits.shape[0]
    total_accepted_fp16 = 0
    total_accepted_int8 = 0
    total_proposed = n_runs * gamma

    for i in range(n_runs):
        for k in range(gamma):
            draft_tok_fp16 = np.argmax(fp16_logits[i, k])
            target_tok = np.argmax(tgt_logits[i, k])
            if draft_tok_fp16 == target_tok:
                total_accepted_fp16 += 1
            else:
                break

        for k in range(gamma):
            draft_tok_int8 = np.argmax(int8_logits[i, k])
            target_tok = np.argmax(tgt_logits[i, k])
            if draft_tok_int8 == target_tok:
                total_accepted_int8 += 1
            else:
                break

    alpha_fp16 = float(total_accepted_fp16) / float(total_proposed)
    alpha_int8 = float(total_accepted_int8) / float(total_proposed)
    delta = alpha_int8 - alpha_fp16

    return {
        "alpha_fp16": alpha_fp16,
        "alpha_int8": alpha_int8,
        "delta": delta
    }
