import random

CONFIGS = [
    {"draft_tokens": [101, 2054, 2003, 1037, 3455, 102], "target_tokens": [101, 2054, 2003, 1037, 3455, 102], "family": "same"},
    {"draft_tokens": [1, 45, 12, 99, 3], "target_tokens": [1, 45, 99, 3], "family": "cross"},
    {"draft_tokens": [5, 10, 15, 20, 25, 30], "target_tokens": [5, 12, 15, 22, 25, 30], "family": "cross"},
]

def align_tokens(draft_tokens, target_tokens):
    n = len(draft_tokens)
    m = len(target_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if draft_tokens[i - 1] == target_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    i, j = n, m
    mapping = []
    while i > 0 and j > 0:
        if draft_tokens[i - 1] == target_tokens[j - 1]:
            mapping.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    mapping.reverse()
    return mapping

def compute_metrics(draft_tokens, target_tokens, mapping, overhead_ms):
    accepted = len(mapping)
    total_draft = max(1, len(draft_tokens))
    acceptance_rate = accepted / total_draft
    effective_throughput = acceptance_rate * (1000.0 / max(0.1, overhead_ms))
    return {
        "acceptance_rate": float(acceptance_rate),
        "overhead_ms": float(overhead_ms),
        "effective_throughput": float(effective_throughput)
    }

def evaluate_uad(config):
    mapping = align_tokens(config["draft_tokens"], config["target_tokens"])
    metrics = compute_metrics(config["draft_tokens"], config["target_tokens"], mapping, 1.5)
    return {
        "mapping": mapping,
        "metrics": metrics,
        "worth_it": config["family"] == "same" or len(mapping) >= 3
    }
