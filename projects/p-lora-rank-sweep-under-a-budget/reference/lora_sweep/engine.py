import ref

def run_baseline(config):
    return ref.simulate_baseline(config)

def run_rank_sweep(ranks, budget_steps):
    return ref.simulate_rank_sweep(ranks, budget_steps)
