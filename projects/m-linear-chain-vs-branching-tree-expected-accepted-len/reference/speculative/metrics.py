def expected_accepted_length_linear(probs):
    exp_len = 0.0
    cum_prob = 1.0
    for p in probs:
        cum_prob *= p
        exp_len += cum_prob
    return exp_len

def expected_accepted_length_tree(parents, probs):
    n = len(parents)
    children = [[] for _ in range(n)]
    for i in range(1, n):
        children[parents[i]].append(i)

    memo = {}

    def get_max_exp(u):
        if u in memo:
            return memo[u]
        p_u = probs[u]
        if not children[u]:
            val = p_u
        else:
            best_child = max(get_max_exp(v) for v in children[u])
            val = p_u * (1.0 + best_child)
        memo[u] = val
        return val

    return get_max_exp(0)
