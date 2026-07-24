import numpy as np


def _oracle(p_teacher: np.ndarray, p_students: np.ndarray):
    t_sorted = np.sort(p_teacher)
    uld = np.array(
        [np.sum(np.abs(t_sorted - np.sort(row))) for row in p_students],
        dtype=np.float64,
    )
    kl = np.array(
        [np.sum(p_teacher * np.log(p_teacher / row)) for row in p_students],
        dtype=np.float64,
    )
    return uld, kl


def _fail():
    return {"rel_err": float("inf"), "nonneg_ok": 0.0, "minimized_at_match": 0.0}


def grade(sol, fx) -> dict:
    p_teacher = fx["uld_p_teacher"]
    p_students = fx["uld_p_students"]
    uld_ref, kl_ref = _oracle(p_teacher, p_students)

    try:
        out = sol.uld_and_kl_along_sweep(p_teacher.copy(), p_students.copy())
    except Exception:
        return _fail()

    try:
        uld_got = np.asarray(out[0], dtype=np.float64).reshape(-1)
        kl_got = np.asarray(out[1], dtype=np.float64).reshape(-1)
    except Exception:
        return _fail()

    if uld_got.shape != uld_ref.shape or kl_got.shape != kl_ref.shape:
        return _fail()
    if not (np.all(np.isfinite(uld_got)) and np.all(np.isfinite(kl_got))):
        return _fail()

    uld_rel = np.abs(uld_got - uld_ref) / (np.abs(uld_ref) + 1e-12)
    kl_rel = np.abs(kl_got - kl_ref) / (np.abs(kl_ref) + 1e-12)
    rel_err = float(max(np.max(uld_rel), np.max(kl_rel)))

    nonneg_ok = 1.0 if (np.all(uld_got >= -1e-9) and np.all(kl_got >= -1e-9)) else 0.0
    # fixture row 0 is the exact teacher==student match (alpha == 0)
    minimized_at_match = (
        1.0
        if (int(np.argmin(uld_got)) == 0 and int(np.argmin(kl_got)) == 0)
        else 0.0
    )

    return {
        "rel_err": rel_err,
        "nonneg_ok": nonneg_ok,
        "minimized_at_match": minimized_at_match,
    }
