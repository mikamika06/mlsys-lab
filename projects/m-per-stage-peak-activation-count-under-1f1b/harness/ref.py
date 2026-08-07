def compute_peak_activations(num_stages, num_microbatches):
    return [min(num_stages - s, num_microbatches) for s in range(num_stages)]

def memory_trace(num_stages, num_microbatches):
    P = num_stages
    M = num_microbatches
    f_done = [[False] * M for _ in range(P)]
    b_done = [[False] * M for _ in range(P)]
    f_ptr = [0] * P
    b_ptr = [0] * P
    stage_active_sets = [set() for _ in range(P)]
    trace = []
    t = 0
    while not all(b_done[P-1]):
        stage_action = [None] * P
        for s in range(P):
            m_b = b_ptr[s]
            can_b = False
            if m_b < M and f_done[s][m_b]:
                if s == P-1 or b_done[s+1][m_b]:
                    can_b = True
            if can_b:
                stage_action[s] = ('B', m_b)
            else:
                m_f = f_ptr[s]
                can_f = False
                if m_f < M:
                    if s == 0 or f_done[s-1][m_f]:
                        current_active = len(stage_active_sets[s])
                        max_allowed = min(P - s, M)
                        if current_active < max_allowed:
                            can_f = True
                if can_f:
                    stage_action[s] = ('F', m_f)
        for s in range(P):
            action = stage_action[s]
            if action is not None:
                kind, m = action
                if kind == 'F':
                    f_done[s][m] = True
                    f_ptr[s] += 1
                    stage_active_sets[s].add(m)
                else:
                    b_done[s][m] = True
                    b_ptr[s] += 1
                    if m in stage_active_sets[s]:
                        stage_active_sets[s].remove(m)
        trace.append([len(stage_active_sets[s]) for s in range(P)])
        t += 1
        if t > 10000:
            break
    return trace

def max_microbatches(num_stages, memory_budget_bytes, activation_bytes_per_mb):
    max_allowed = memory_budget_bytes // activation_bytes_per_mb
    return max(0, max_allowed)

TEST_CASES = [
    (4, 8),
    (2, 4),
    (8, 16),
    (4, 3),
]
