def build_head_map(num_q_heads, num_kv_heads):
    group = num_q_heads // num_kv_heads
    return [q // group for q in range(num_q_heads)]


def build_query_groups(num_q_heads, num_kv_heads):
    head_map = build_head_map(num_q_heads, num_kv_heads)
    groups = [[] for _ in range(num_kv_heads)]
    for q, k in enumerate(head_map):
        groups[k].append(q)
    return groups
