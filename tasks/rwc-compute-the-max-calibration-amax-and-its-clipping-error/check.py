import numpy as np

def _oracle(batches):
    all_vals = np.concatenate([b.ravel() for b in batches])
    amax = float(np.max(np.abs(all_vals)))
    scale = amax / 127.0
    recon_all = []
    for b in batches:
        q = np.round(b / scale)
        q = np.clip(q, -127, 127)
        recon = q * scale
        recon_all.append(recon.ravel())
    recon_vals = np.concatenate(recon_all)
    mse = float(np.mean((recon_vals - all_vals)**2))
    return amax, mse

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    b1 = rng.uniform(-5, 5, size=(4,3)).astype(np.float64)
    b2 = rng.uniform(-5, 5, size=(6,2)).astype(np.float64)
    b3 = rng.uniform(-5, 5, size=(5,5)).astype(np.float64)
    batches = [b1,b2,b3]
    oracle_amax, oracle_mse = _oracle(batches)

    try:
        student_amax, student_mse = sol.calibrate_max_and_error(batches)
    except Exception:
        return {"rel_err": 1.0}

    rel_amax = abs(student_amax - oracle_amax) / (oracle_amax + 1e-12)
    rel_mse  = abs(student_mse  - oracle_mse ) / (oracle_mse  + 1e-12)
    rel_err = max(rel_amax, rel_mse)

    return {"rel_err": rel_err}
