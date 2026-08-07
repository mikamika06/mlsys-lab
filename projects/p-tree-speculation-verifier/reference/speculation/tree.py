import numpy as np


def build_tree(paths):
    tokens = []
    parents = []
    node_map = {}
    for path in paths:
        parent_id = -1
        for token in path:
            key = (parent_id, token)
            if key not in node_map:
                node_id = len(tokens)
                tokens.append(token)
                parents.append(parent_id)
                node_map[key] = node_id
            parent_id = node_map[key]
    return tokens, parents


def tree_attention_mask(parents):
    n = len(parents)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        curr = i
        while curr != -1:
            mask[i, curr] = True
            curr = parents[curr]
    return mask


def verify_tree(tokens, parents, draft_probs, target_probs, node_r, resample_r):
    accepted = []
    u = -1
    children = [[] for _ in range(len(parents) + 1)]
    for i, p in enumerate(parents):
        children[p + 1].append(i)

    while True:
        P = target_probs[u + 1]
        C_u = children[u + 1]

        if not C_u:
            u_res = resample_r[u + 1]
            cdf = 0.0
            for v, p_val in enumerate(P):
                cdf += p_val
                if cdf > u_res:
                    accepted.append(v)
                    break
            else:
                accepted.append(len(P) - 1)
            break

        m_vals = []
        q_draft = [0.0] * len(P)
        for i in C_u:
            tok = tokens[i]
            q = draft_probs[i]
            q_draft[tok] = q
            m_vals.append(min(P[tok], q))

        sum_m = sum(m_vals)
        u_val = node_r[u + 1]

        if u_val < sum_m:
            cdf = 0.0
            acc_i = C_u[-1]
            for idx, (i, m) in enumerate(zip(C_u, m_vals)):
                cdf += m
                if u_val < cdf:
                    acc_i = i
                    break
            accepted.append(tokens[acc_i])
            u = acc_i
        else:
            R = [max(0.0, P[v] - q_draft[v]) for v in range(len(P))]
            Z = sum(R)
            if Z > 0:
                P_prime = [r / Z for r in R]
            else:
                P_prime = [1.0 / len(P)] * len(P)

            u_res = resample_r[u + 1]
            cdf = 0.0
            for v, p_val in enumerate(P_prime):
                cdf += p_val
                if cdf > u_res:
                    accepted.append(v)
                    break
            else:
                accepted.append(len(P) - 1)
            break

    return accepted


def expected_length(tokens, parents, draft_probs, target_probs):
    n = len(parents)
    E = [0.0] * n
    for i in range(n - 1, -1, -1):
        P = target_probs[i + 1]
        children = [j for j, p in enumerate(parents) if p == i]
        expected_sum = 0.0
        for j in children:
            tok = tokens[j]
            q = draft_probs[j]
            m = min(P[tok], q)
            expected_sum += m * E[j]
        E[i] = 1.0 + expected_sum

    P = target_probs[0]
    root_children = [j for j, p in enumerate(parents) if p == -1]
    expected_sum = 0.0
    for j in root_children:
        tok = tokens[j]
        q = draft_probs[j]
        m = min(P[tok], q)
        expected_sum += m * E[j]
    return 1.0 + expected_sum
