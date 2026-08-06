import numpy as np


def generate_trace_data(seed=42):
    rng = np.random.default_rng(seed)
    traces = []
    for _ in range(100):
        length = rng.integers(1, 8)
        trace = []
        for pos in range(length):
            prob = max(0.1, 0.95 - 0.12 * pos)
            acc = bool(rng.random() < prob)
            trace.append(acc)
            if not acc:
                break
        traces.append(trace)
    return traces


def generate_domain_curves():
    return {
        "code": np.array([0.95, 0.90, 0.82, 0.70, 0.55]),
        "chat": np.array([0.88, 0.75, 0.60, 0.40, 0.20]),
        "summarization": np.array([0.92, 0.85, 0.78, 0.68, 0.58]),
    }


def generate_prob_distributions(seed=123):
    rng = np.random.default_rng(seed)
    vocab_size = 100
    p_draft = rng.dirichlet(np.ones(vocab_size))
    p_target = rng.dirichlet(np.ones(vocab_size))
    return p_draft, p_target


def compute_acceptance_metrics(traces):
    total_accepted = 0
    total_drafted = 0
    max_len = max(len(t) for t in traces) if traces else 0
    pos_accepted = np.zeros(max_len, dtype=np.float64)
    pos_counts = np.zeros(max_len, dtype=np.float64)

    for trace in traces:
        for pos, accepted in enumerate(trace):
            total_drafted += 1
            pos_counts[pos] += 1
            if accepted:
                total_accepted += 1
                pos_accepted[pos] += 1

    flat_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
    decay_curve = np.zeros(max_len, dtype=np.float64)
    valid_mask = pos_counts > 0
    decay_curve[valid_mask] = pos_accepted[valid_mask] / pos_counts[valid_mask]

    return float(flat_rate), decay_curve


def analyze_domain_acceptance(domain_decay_curves, draft_length):
    results = {}
    for domain, curve in domain_decay_curves.items():
        k = min(draft_length, len(curve))
        if k == 0:
            results[domain] = 0.0
            continue
        sub_curve = np.asarray(curve[:k], dtype=np.float64)
        prefix_probs = np.cumprod(sub_curve)
        expected_accepted = np.sum(prefix_probs)
        results[domain] = float(expected_accepted / k)
    return results


def kl_divergence_to_acceptance_bound(p_draft, p_target):
    p_draft = np.asarray(p_draft, dtype=np.float64)
    p_target = np.asarray(p_target, dtype=np.float64)

    p_draft = np.clip(p_draft, 1e-12, 1.0)
    p_draft = p_draft / np.sum(p_draft)

    p_target = np.clip(p_target, 1e-12, 1.0)
    p_target = p_target / np.sum(p_target)

    kl_div = np.sum(p_draft * np.log(p_draft / p_target))
    tv_dist = 0.5 * np.sum(np.abs(p_draft - p_target))
    lower_bound = max(0.0, float(1.0 - tv_dist))

    return float(kl_div), lower_bound
