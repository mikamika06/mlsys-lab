import numpy as np

def separate_failures(attention_weights, token_ids):
    attn_mean = float(np.mean(attention_weights))
    token_validity = float(np.mean(token_ids > 0))
    if attn_mean < 0.1 and token_validity > 0.9:
        return "attention_failure"
    elif token_validity <= 0.9:
        return "tokenization_failure"
    return "healthy"

def compare_methods(tasks, method_a_scores, method_b_scores):
    return {
        "method_a_mean": float(np.mean(method_a_scores)),
        "method_b_mean": float(np.mean(method_b_scores)),
        "superior": "method_b" if np.mean(method_b_scores) > np.mean(method_a_scores) else "method_a"
    }
