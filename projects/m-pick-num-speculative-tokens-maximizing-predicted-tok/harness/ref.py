import numpy as np

MODELS = [
    {"name": "model_a", "base_latency": 10.0, "draft_latency": 2.0},
    {"name": "model_b", "base_latency": 15.0, "draft_latency": 3.0},
    {"name": "model_c", "base_latency": 20.0, "draft_latency": 2.5},
    {"name": "model_d", "base_latency": 25.0, "draft_latency": 4.0},
]

CODE_EDIT_TRACE = [
    "def add(a, b):",
    "    return a + b",
    "def sub(a, b):",
    "    return a - b",
    "def add(a, b):",
    "    return a + b",
    "def mul(a, b):",
    "    return a * b"
]

CONCURRENCY_LEVELS = [1, 4, 16, 64]

def compute_predicted_tokens_per_sec(model, k, alpha):
    base = model["base_latency"]
    draft = model["draft_latency"]
    expected_accepted = sum((alpha ** i) for i in range(1, k + 1))
    total_latency = base + k * draft
    return (1.0 + expected_accepted) / total_latency

def get_optimal_k(model, alpha, max_k=5):
    best_k = 1
    best_val = -1.0
    for k in range(1, max_k + 1):
        val = compute_predicted_tokens_per_sec(model, k, alpha)
        if val > best_val:
            best_val = val
            best_k = k
    return best_k

def evaluate_ngram_acceptance(trace, n=2, k=3):
    tokens = []
    for line in trace:
        tokens.extend(line.split())
    if len(tokens) < n + k:
        return 0.5
    accepted = 0
    total = 0
    for i in range(len(tokens) - n - k + 1):
        context = tuple(tokens[i:i+n])
        draft = tokens[i+n:i+n+k]
        target_next = tokens[i+n]
        total += 1
        if draft and draft[0] == target_next:
            accepted += 1
    if total == 0:
        return 0.0
    return float(accepted) / float(total)

def decide_go_no_go(concurrency, acceptance_rate, model):
    threshold = 0.4 + 0.05 * (concurrency / 64.0)
    if acceptance_rate >= threshold and model["base_latency"] > 5.0:
        return True
    return False
