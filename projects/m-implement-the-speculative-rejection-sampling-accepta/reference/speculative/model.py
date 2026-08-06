def expected_tokens(alphas):
    ans = 0.0
    p = 1.0
    for a in alphas:
        p *= a
        ans += p
    return ans

def compute_speedup(expected_n, k, draft_cost_ratio):
    return (expected_n + 1.0) / (k * draft_cost_ratio + 1.0)
