import numpy as np
from mlsys import scorers, probe

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(4)
    A = rng.standard_normal((30, 15))
    ref = (np.sum(A*A, axis=1).reshape(-1,1) +
           np.sum(A*A, axis=1).reshape(1,-1) -
           2 * A.dot(A.T))
    op_count = probe.count_line_events(sol.pairwise_sq_dists, A)
    rel_err_val = scorers.rel_err(ref, sol.pairwise_sq_dists(A))
    return {"op_count": op_count, "rel_err": rel_err_val}
