import math


def reconstruct_output(states):
    S = len(states)
    num_queries = len(states[0][0])
    d_v = len(states[0][2][0])

    m = [[float(val) for val in states[s][0]] for s in range(S)]
    l = [[float(val) for val in states[s][1]] for s in range(S)]
    o = [[[float(val) for val in row] for row in states[s][2]] for s in range(S)]

    global_m = [0.0] * num_queries
    for q_idx in range(num_queries):
        max_val = m[0][q_idx]
        for s_idx in range(1, S):
            if m[s_idx][q_idx] > max_val:
                max_val = m[s_idx][q_idx]
        global_m[q_idx] = max_val

    scale = [[0.0] * num_queries for _ in range(S)]
    for s_idx in range(S):
        for q_idx in range(num_queries):
            scale[s_idx][q_idx] = math.exp(m[s_idx][q_idx] - global_m[q_idx])

    contributions = [[0.0] * num_queries for _ in range(S)]
    for s_idx in range(S):
        for q_idx in range(num_queries):
            contributions[s_idx][q_idx] = scale[s_idx][q_idx] * l[s_idx][q_idx]

    denom = [0.0] * num_queries
    for q_idx in range(num_queries):
        acc = 0.0
        for s_idx in range(S):
            acc += contributions[s_idx][q_idx]
        denom[q_idx] = acc

    output = [[0.0] * d_v for _ in range(num_queries)]
    for q_idx in range(num_queries):
        for d_idx in range(d_v):
            acc = 0.0
            for s_idx in range(S):
                acc += scale[s_idx][q_idx] * o[s_idx][q_idx][d_idx]
            output[q_idx][d_idx] = acc / denom[q_idx]

    global_lse = [0.0] * num_queries
    for q_idx in range(num_queries):
        global_lse[q_idx] = global_m[q_idx] + math.log(denom[q_idx])

    rank_mass = [[0.0] * num_queries for _ in range(S)]
    for s_idx in range(S):
        for q_idx in range(num_queries):
            rank_mass[s_idx][q_idx] = contributions[s_idx][q_idx] / denom[q_idx]

    return output, global_lse, rank_mass
