import numpy as np
from typing import Tuple

def _softmax(z: np.ndarray, T: float) -> np.ndarray:
    z = z / T
    z_max = np.max(z)
    e = np.exp(z - z_max)
    return e / np.sum(e)

def _kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

def grade(sol, fx) -> dict:
    max_rel_err = 0.0
    for _ in range(5):
        n = np.random.randint(2, 10)
        teacher_logits = np.random.randn(n).astype(np.float64)
        student_logits = np.random.randn(n).astype(np.float64)
        temperature = np.random.uniform(0.5, 5.0)
        try:
            got = sol.kl_divergences(teacher_logits, student_logits, temperature)
        except Exception:
            return {"rel_err": 1.0}
        if not isinstance(got, (tuple, list)) or len(got) != 2:
            return {"rel_err": 1.0}
        forward_ref = _kl(_softmax(teacher_logits, temperature),
                          _softmax(student_logits, temperature))
        reverse_ref = _kl(_softmax(student_logits, temperature),
                          _softmax(teacher_logits, temperature))
        ref = np.array([forward_ref, reverse_ref], dtype=np.float64)
        got_arr = np.array(got, dtype=np.float64)
        rel_err = np.linalg.norm(got_arr - ref) / (np.linalg.norm(ref) + 1e-12)
        if rel_err > max_rel_err:
            max_rel_err = rel_err
    return {"rel_err": max_rel_err}
