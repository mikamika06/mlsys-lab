def measure_acceptance(trace):
    raise NotImplementedError

def compute_speedup(p, gamma, t_draft, t_target, t_verify):
    raise NotImplementedError

def batch_speedup_table(p, gamma, max_b, cost_model):
    raise NotImplementedError

def get_cutoff_batch_size(p, gamma, max_b, cost_model):
    raise NotImplementedError
