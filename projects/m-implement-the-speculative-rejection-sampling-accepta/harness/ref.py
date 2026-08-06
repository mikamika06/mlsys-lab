import numpy as np

def generate_fixtures():
    np.random.seed(42)
    fixtures = []
    for _ in range(50):
        k, v = 4, 32
        tp = np.random.dirichlet(np.ones(v), k)
        dp = np.random.dirichlet(np.ones(v), k)
        tokens = np.array([np.random.choice(v, p=dp[i]) for i in range(k)])
        u = np.random.rand(k)
        fixtures.append((tp, dp, tokens, u))
    return fixtures

def evaluate_draft(target_p, draft_p, tokens, u):
    k = target_p.shape[0]
    for i in range(k):
        p_i = target_p[i, tokens[i]]
        q_i = draft_p[i, tokens[i]]
        if u[i] >= (p_i / q_i):
            diff = np.maximum(0.0, target_p[i] - draft_p[i])
            s = np.sum(diff)
            if s > 0:
                diff /= s
            return i, diff
    return k, None

def expected_tokens(alphas):
    ans = 0.0
    p = 1.0
    for a in alphas:
        p *= a
        ans += p
    return ans

def compute_speedup(expected_n, k, draft_cost_ratio):
    return (expected_n + 1.0) / (k * draft_cost_ratio + 1.0)
