import numpy as np


def generate_pairings_data(seed=42):
    rng = np.random.RandomState(seed)
    pairings = {}
    names = ["pair_alpha", "pair_beta", "pair_gamma"]
    for name in names:
        length = 10
        draft = rng.randint(0, 50, size=length).tolist()
        target = list(draft)
        mismatch_idx = rng.randint(1, length)
        target[mismatch_idx] = target[mismatch_idx] + 100
        pairings[name] = {"draft_tokens": draft, "target_tokens": target}
    return pairings


PAIRINGS_DATA = generate_pairings_data()


def generate_tokenizer_cases():
    return [
        {"draft": ["a", "b", "c"], "target": ["a", "b", "c"], "expected": "identical"},
        {"draft": ["a", "b"], "target": ["a", "b", "c"], "expected": "compatible_subset"},
        {"draft": ["a", "x"], "target": ["a", "b", "c"], "expected": "cross_tokenizer"},
    ]


TOKENIZER_CASES = generate_tokenizer_cases()


def generate_selection_candidates():
    return [
        {"name": "draft_fast_low_acc", "draft_latency": 1.5, "acceptance_rate": 0.4},
        {"name": "draft_med_med_acc", "draft_latency": 3.0, "acceptance_rate": 0.75},
        {"name": "draft_slow_high_acc", "draft_latency": 6.0, "acceptance_rate": 0.95},
    ]


SELECTION_CANDIDATES = generate_selection_candidates()
TARGET_LATENCY = 25.0
GAMMA = 5


def compute_acceptance_rate(draft_tokens, target_tokens):
    if not draft_tokens:
        return 0.0
    accepted = 0
    n = min(len(draft_tokens), len(target_tokens))
    for i in range(n):
        if draft_tokens[i] == target_tokens[i]:
            accepted += 1
        else:
            break
    return accepted / len(draft_tokens)


def evaluate_pairings(pairings_data):
    results = {}
    for name, data in pairings_data.items():
        rate = compute_acceptance_rate(data["draft_tokens"], data["target_tokens"])
        results[name] = rate
    return results


def classify_tokenizer_compatibility(draft_vocab, target_vocab):
    if draft_vocab == target_vocab:
        return "identical"
    draft_set = set(draft_vocab)
    target_set = set(target_vocab)
    if draft_set.issubset(target_set):
        return "compatible_subset"
    return "cross_tokenizer"


def compute_expected_throughput(gamma, acceptance_rate, draft_latency, target_latency):
    expected_accepted = acceptance_rate * gamma
    expected_tokens = 1.0 + expected_accepted
    iteration_latency = gamma * draft_latency + target_latency
    return expected_tokens / iteration_latency


def select_optimal_draft(candidates, target_latency, gamma):
    best_name = None
    best_throughput = -1.0
    for cand in candidates:
        tp = compute_expected_throughput(
            gamma, cand["acceptance_rate"], cand["draft_latency"], target_latency
        )
        if tp > best_throughput:
            best_throughput = tp
            best_name = cand["name"]
    return {"best_draft": best_name, "throughput": best_throughput}
