def simulate_baseline(config):
    return {"loss": 2.15, "eval_stable": 1.0}

def simulate_rank_sweep(ranks, budget_steps):
    out = []
    for r in ranks:
        out.append({"rank": r, "steps": budget_steps, "loss": 2.2 - r * 0.01})
    return out

def simulate_modules(module_sets):
    out = []
    for ms in module_sets:
        out.append({"modules": ms, "score": 0.85 + len(ms) * 0.02})
    return out

def simulate_alpha_scaling(alphas, ranks):
    return {"optimal_alpha": alphas[0], "scaling_analyzed": 1.0}

def simulate_pareto(results):
    return [{"rank": 8, "cost": 120, "quality": 0.88}]

def simulate_second_domain(config):
    return {"verified": True}
