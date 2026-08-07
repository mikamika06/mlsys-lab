import numpy as np
from gkdstep.loss import softmax


def compute_tv_distance(p_probs, q_probs):
    """Compute total variation distance between probability distributions."""
    return 0.5 * np.sum(np.abs(p_probs - q_probs), axis=-1)


def measure_sequence_drift(teacher_seq_logits, student_seq_logits, beta=0.5, temperature=1.0):
    """Measure cumulative distribution drift over autoregressive time steps."""
    p = softmax(teacher_seq_logits, temperature)
    q = softmax(student_seq_logits, temperature)
    step_tvd = compute_tv_distance(p, q)
    step_tvd_mean = np.mean(step_tvd, axis=0)

    seq_len = step_tvd_mean.shape[0]
    c_t = np.zeros(seq_len, dtype=np.float64)
    prefix_sum = 0.0
    for t in range(seq_len):
        c_t[t] = step_tvd_mean[t] + float(beta) * prefix_sum
        prefix_sum += step_tvd_mean[t]

    return {
        "step_drifts": c_t,
        "mean_drift": float(np.mean(c_t)),
        "off_policy_baseline": float(np.mean(step_tvd_mean)),
    }
