import numpy as np

np.random.seed(42)

H_VALID = np.eye(4, dtype=np.float32) * 2.0
H_NAN = np.eye(4, dtype=np.float32)
H_NAN[1, 2] = np.nan

H_INF = np.eye(4, dtype=np.float32)
H_INF[0, 0] = np.inf

TEST_HESSIANS = [
    {"matrix": H_VALID, "valid": True},
    {"matrix": H_NAN, "valid": False},
    {"matrix": H_INF, "valid": False},
]

DECISION_CASES = [
    {"hessian": H_VALID, "condition_number": 10.0, "has_nan": False, "model_free_ptq": False},
    {"hessian": H_NAN, "condition_number": 1.0, "has_nan": True, "model_free_ptq": True},
    {"hessian": H_VALID, "condition_number": 1e8, "has_nan": False, "model_free_ptq": True},
]

def check_hessian(h):
    if not isinstance(h, np.ndarray):
        h = np.array(h, dtype=np.float32)
    if not np.isfinite(h).all():
        return False
    return True

def require_model_free_ptq(h, cond_thresh=1e6):
    if not check_hessian(h):
        return True
    try:
        cond = np.linalg.cond(h)
        if np.isnan(cond) or cond > cond_thresh:
            return True
    except Exception:
        return True
    return False
