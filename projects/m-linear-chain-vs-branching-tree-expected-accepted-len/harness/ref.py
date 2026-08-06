import numpy as np

def make_linear_case(length, seed=42):
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.1, 0.95, size=length).tolist()
    return {"probs": probs}

def make_tree_case(num_nodes=10, seed=42):
    rng = np.random.default_rng(seed)
    parents = [-1]
    for i in range(1, num_nodes):
        parents.append(int(rng.integers(0, i)))
    probs = rng.uniform(0.1, 0.95, size=num_nodes).tolist()
    return {"parents": parents, "probs": probs}

TEST_LINEAR_CASES = [make_linear_case(k, seed=10 + k) for k in range(1, 8)]
TEST_TREE_CASES = [make_tree_case(n, seed=100 + n) for n in range(2, 15)]

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

def verify_tree_sample(parents, accepts):
    n = len(parents)
    accepted_set = {i for i, acc in enumerate(accepts) if acc}

    if 0 not in accepted_set:
        return []

    valid_nodes = set()
    for i in range(n):
        if i in accepted_set:
            curr = i
            path = []
            possible = True
            while curr != -1:
                if curr not in accepted_set:
                    possible = False
                    break
                path.append(curr)
                curr = parents[curr]
            if possible:
                valid_nodes.add(i)

    children = [[] for _ in range(n)]
    for i in range(1, n):
        if i in valid_nodes:
            p = parents[i]
            if p in valid_nodes:
                children[p].append(i)

    best_path = []

    def dfs(u, current_path):
        nonlocal best_path
        current_path.append(u)
        if len(current_path) > len(best_path):
            best_path = list(current_path)
        for v in children[u]:
            dfs(v, current_path)
        current_path.pop()

    dfs(0, [])
    return best_path
