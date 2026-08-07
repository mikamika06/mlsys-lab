import sys
import ref

def expected_cutoff(p, gamma, max_b, cost_model):
    best = 0
    for b in range(1, max_b + 1):
        td, tt, tv = cost_model(b, gamma)
        if p >= 1.0: e = 1.0 + gamma
        else: e = 1.0 + (p - p**(gamma+1))/(1.0-p)
        if e * tt / (gamma*td + tv) > 1.0:
            best = b
        else:
            break
    return best

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        import specdec.analyzer as an
    except ImportError:
        return {"cutoff_ok": 0.0, "cutoff_zero_ok": 0.0}

    m = {"cutoff_ok": 0.0, "cutoff_zero_ok": 0.0}
    try:
        c1 = an.get_cutoff_batch_size(0.8, 4, 32, ref.cost_model)
        if c1 == expected_cutoff(0.8, 4, 32, ref.cost_model):
            m["cutoff_ok"] = 1.0

        c2 = an.get_cutoff_batch_size(0.1, 4, 32, ref.cost_model)
        if c2 == expected_cutoff(0.1, 4, 32, ref.cost_model):
            m["cutoff_zero_ok"] = 1.0
    except Exception:
        pass
    return m
