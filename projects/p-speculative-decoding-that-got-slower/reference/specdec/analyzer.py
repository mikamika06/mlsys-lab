def measure_acceptance(trace):
    d = sum(x["drafted"] for x in trace)
    a = sum(x["accepted"] for x in trace)
    return a / d if d > 0 else 0.0

def compute_speedup(p, gamma, t_draft, t_target, t_verify):
    if p >= 1.0:
        e_toks = 1.0 + gamma
    else:
        e_toks = 1.0 + (p - p**(gamma + 1)) / (1.0 - p)

    t_spec = gamma * t_draft + t_verify
    t_base = e_toks * t_target
    return t_base / t_spec

def batch_speedup_table(p, gamma, max_b, cost_model):
    res = []
    for b in range(1, max_b + 1):
        td, tt, tv = cost_model(b, gamma)
        res.append(compute_speedup(p, gamma, td, tt, tv))
    return res

def get_cutoff_batch_size(p, gamma, max_b, cost_model):
    best_b = 0
    for b in range(1, max_b + 1):
        td, tt, tv = cost_model(b, gamma)
        if compute_speedup(p, gamma, td, tt, tv) > 1.0:
            best_b = b
        else:
            break
    return best_b
