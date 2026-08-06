import numpy as np


def rejection_sample(target_probs, draft_probs, draft_tokens, rng):
    k = len(draft_tokens)
    accepted_tokens = []
    num_accepted = 0

    for i in range(k):
        t = int(draft_tokens[i])
        p = float(target_probs[i, t])
        q = float(draft_probs[i, t])

        ratio = p / q if q > 0 else (1.0 if p > 0 else 0.0)
        u = float(rng.uniform(0.0, 1.0))

        if u <= min(1.0, ratio):
            accepted_tokens.append(t)
            num_accepted += 1
        else:
            p_prime = np.maximum(0.0, target_probs[i] - draft_probs[i])
            total = float(np.sum(p_prime))
            if total > 0:
                p_prime = p_prime / total
            else:
                p_prime = target_probs[i] / np.sum(target_probs[i])

            resampled = int(rng.choice(len(p_prime), p=p_prime))
            accepted_tokens.append(resampled)
            break

    if num_accepted == k:
        extra = int(rng.choice(target_probs.shape[1], p=target_probs[k]))
        accepted_tokens.append(extra)

    return np.array(accepted_tokens, dtype=np.int64), num_accepted


def compute_speedup(acceptance_trace, k, draft_cost, target_cost):
    trace = np.asarray(acceptance_trace, dtype=np.float64)
    total_passes = len(trace)
    if total_passes == 0:
        return {
            "realized_speedup": 0.0,
            "theoretical_speedup": 0.0,
            "mean_acceptance_rate": 0.0,
            "expected_tokens_per_pass": 0.0,
        }

    total_accepted_draft = float(np.sum(trace))
    mean_accepted = total_accepted_draft / total_passes
    alpha = total_accepted_draft / (total_passes * k)

    expected_tokens_per_pass = mean_accepted + 1.0
    c = draft_cost / target_cost

    realized_speedup = expected_tokens_per_pass / (1.0 + k * c)

    if abs(alpha - 1.0) < 1e-9:
        theoretical_tokens = float(k + 1)
    else:
        theoretical_tokens = (1.0 - alpha ** (k + 1)) / (1.0 - alpha)

    theoretical_speedup = theoretical_tokens / (1.0 + k * c)

    return {
        "realized_speedup": float(realized_speedup),
        "theoretical_speedup": float(theoretical_speedup),
        "mean_acceptance_rate": float(alpha),
        "expected_tokens_per_pass": float(expected_tokens_per_pass),
    }


def find_optimal_k(alpha, c, max_k):
    best_k = 1
    best_speedup = -1.0
    speedups = {}

    for k in range(1, max_k + 1):
        if abs(alpha - 1.0) < 1e-9:
            tokens = float(k + 1)
        else:
            tokens = (1.0 - alpha ** (k + 1)) / (1.0 - alpha)

        speedup = tokens / (1.0 + k * c)
        speedups[k] = float(speedup)

        if speedup > best_speedup + 1e-12:
            best_speedup = speedup
            best_k = k

    return best_k, speedups


def get_m1_dist_fixture():
    p0 = np.array([0.5, 0.3, 0.1, 0.1])
    q0 = np.array([0.2, 0.6, 0.1, 0.1])
    target_probs = np.vstack([p0, p0])
    draft_probs = np.vstack([q0])
    return target_probs, draft_probs, p0, q0


M2_TRACES = [
    {"trace": [2, 2, 2, 2, 2], "k": 2, "draft_cost": 0.1, "target_cost": 1.0},
    {"trace": [0, 1, 2, 1, 0, 2], "k": 3, "draft_cost": 0.05, "target_cost": 1.0},
    {"trace": [4, 4, 3, 2, 4, 1, 0], "k": 4, "draft_cost": 0.08, "target_cost": 1.0},
]

M3_CASES = [
    {"alpha": 0.8, "c": 0.05, "max_k": 8},
    {"alpha": 0.5, "c": 0.2, "max_k": 5},
    {"alpha": 0.95, "c": 0.01, "max_k": 12},
]
