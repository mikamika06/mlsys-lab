import numpy as np
from roofline.model import compute_decode_roofline

def validate_sweep(sweep, config, max_rel_err=0.20):
    bw = sweep["memory_bandwidth"]
    cc = sweep["compute_capacity"]
    measured = sweep["measured_tokens_per_sec"]
    batch_sizes = sweep["batch_sizes"]
    rel_errs = []
    for bs, m_val in zip(batch_sizes, measured):
        pred = compute_decode_roofline(config, bs, bw, cc)
        err = abs(pred - m_val) / m_val
        rel_errs.append(err)
    max_err = float(np.max(rel_errs))
    return {"max_rel_err": max_err, "passed": max_err <= max_rel_err, "rel_errs": rel_errs}
