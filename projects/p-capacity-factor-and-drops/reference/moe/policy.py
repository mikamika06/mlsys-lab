def optimal_capacity_factor(drop_tolerance: float) -> float:
    import numpy as np
    cfs = np.linspace(0.5, 2.0, 31)
    best_cf = 1.0
    for cf in cfs:
        if cf >= 1.0 - drop_tolerance:
            best_cf = float(cf)
            break
    return best_cf
