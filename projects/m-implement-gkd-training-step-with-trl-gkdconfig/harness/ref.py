from dataclasses import dataclass
import numpy as np


@dataclass
class GKDConfig:
    temperature: float = 1.0
    lmbda: float = 0.5
    beta: float = 0.5
    divergence_type: str = "forward_kl"
    max_new_tokens: int = 16


def softmax(logits, temperature=1.0):
    scaled = logits / float(temperature)
    scaled -= np.max(scaled, axis=-1, keepdims=True)
    exp_l = np.exp(scaled)
    return exp_l / np.sum(exp_l, axis=-1, keepdims=True)


def compute_divergence(teacher_logits, student_logits, divergence_type="forward_kl", temperature=1.0):
    p = softmax(teacher_logits, temperature)
    q = softmax(student_logits, temperature)
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)

    if divergence_type == "forward_kl":
        div = np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=-1)
    elif divergence_type == "reverse_kl":
        div = np.sum(q_safe * (np.log(q_safe) - np.log(p_safe)), axis=-1)
    elif divergence_type == "jsd":
        m = 0.5 * (p_safe + q_safe)
        kl_pm = np.sum(p_safe * (np.log(p_safe) - np.log(m)), axis=-1)
        kl_qm = np.sum(q_safe * (np.log(q_safe) - np.log(m)), axis=-1)
        div = 0.5 * kl_pm + 0.5 * kl_qm
    else:
        raise ValueError(f"Unsupported divergence_type: {divergence_type}")

    return float(np.mean(div) * (float(temperature) ** 2))


def compute_gkd_step_loss(teacher_logits, student_logits, config):
    return compute_divergence(
        teacher_logits,
        student_logits,
        divergence_type=config.divergence_type,
        temperature=config.temperature,
    )


def compute_tv_distance(p_probs, q_probs):
    return 0.5 * np.sum(np.abs(p_probs - q_probs), axis=-1)


def measure_sequence_drift(teacher_seq_logits, student_seq_logits, beta=0.5, temperature=1.0):
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


def generate_test_cases():
    rng = np.random.RandomState(42)
    cases = []
    for _ in range(6):
        t_logits = rng.randn(4, 16, 32)
        s_logits = rng.randn(4, 16, 32) + 0.3
        cases.append((t_logits, s_logits))
    return cases


TEST_CASES = generate_test_cases()
