import numpy as np
from typing import Dict, Any


def compute_kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    p_logits = p_logits - np.max(p_logits, axis=-1, keepdims=True)
    q_logits = q_logits - np.max(q_logits, axis=-1, keepdims=True)

    p_exp = np.exp(p_logits)
    p_probs = p_exp / np.sum(p_exp, axis=-1, keepdims=True)
    p_logprobs = p_logits - np.log(np.sum(p_exp, axis=-1, keepdims=True))

    q_exp = np.exp(q_logits)
    q_logprobs = q_logits - np.log(np.sum(q_exp, axis=-1, keepdims=True))

    kl = np.sum(p_probs * (p_logprobs - q_logprobs), axis=-1)
    return np.maximum(kl, 0.0)


def compute_flip_rate(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    p_top = np.argmax(p_logits, axis=-1)
    q_top = np.argmax(q_logits, axis=-1)
    return float(np.mean(p_top != q_top))


def compute_perplexity_from_logprobs(log_probs: np.ndarray) -> float:
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.size == 0:
        return 1.0
    mean_neg_logprob = -np.mean(log_probs)
    return float(np.exp(mean_neg_logprob))


def compute_recovery_percentage(
    baseline_score: float,
    quantized_score: float,
    random_score: float
) -> float:
    denom = baseline_score - random_score
    if abs(denom) < 1e-12:
        return 100.0 if quantized_score >= baseline_score else 0.0
    rec = ((quantized_score - random_score) / denom) * 100.0
    return float(rec)


def parse_lm_eval_recovery(
    eval_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    random_baseline: float = 0.0
) -> Dict[str, float]:
    eval_results_data = eval_results.get("results", {})
    base_results_data = baseline_results.get("results", {})

    out = {}
    for task_name, task_data in eval_results_data.items():
        if task_name not in base_results_data:
            continue

        q_metric = None
        b_metric = None

        for k in ("acc_norm,none", "acc,none", "acc_norm", "acc", "exact_match,none", "exact_match"):
            if k in task_data and k in base_results_data[task_name]:
                q_metric = task_data[k]
                b_metric = base_results_data[task_name][k]
                break

        if q_metric is None:
            continue

        rec = compute_recovery_percentage(b_metric, q_metric, random_baseline)
        out[task_name] = rec

    return out


def generate_test_data():
    rng = np.random.RandomState(1337)

    p_logits = rng.randn(4, 32, 256)
    q_logits = p_logits + rng.normal(0, 0.5, size=p_logits.shape)

    seq_log_probs = -rng.exponential(scale=1.5, size=128)

    baseline_eval = {
        "results": {
            "hellaswag": {"acc_norm,none": 0.78, "acc,none": 0.62},
            "arc_challenge": {"acc_norm,none": 0.52, "acc,none": 0.48},
            "gsm8k": {"exact_match,none": 0.35}
        }
    }

    quantized_eval = {
        "results": {
            "hellaswag": {"acc_norm,none": 0.74, "acc,none": 0.59},
            "arc_challenge": {"acc_norm,none": 0.49, "acc,none": 0.45},
            "gsm8k": {"exact_match,none": 0.28}
        }
    }

    return p_logits, q_logits, seq_log_probs, baseline_eval, quantized_eval
