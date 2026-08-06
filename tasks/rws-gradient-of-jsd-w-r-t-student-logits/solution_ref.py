import math
import numpy as np


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    
    max_val = z[0]
    for i in range(1, n):
        if z[i] > max_val:
            max_val = z[i]
            
    e = np.empty(n, dtype=np.float64)
    sum_e = 0.0
    for i in range(n):
        val = math.exp(z[i] - max_val)
        e[i] = val
        sum_e += val
        
    return e / sum_e


def jsd_grad_wrt_student_logits(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    beta: float,
) -> np.ndarray:
    p = _softmax(teacher_logits)
    q = _softmax(student_logits)
    n = p.shape[0]
    
    m = np.empty(n, dtype=np.float64)
    for i in range(n):
        m[i] = beta * p[i] + (1.0 - beta) * q[i]
        
    g = np.empty(n, dtype=np.float64)
    for i in range(n):
        g[i] = (1.0 - beta) * math.log(q[i] / m[i])
        
    sum_qg = 0.0
    for i in range(n):
        sum_qg += q[i] * g[i]
        
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = q[i] * (g[i] - sum_qg)
        
    return result
