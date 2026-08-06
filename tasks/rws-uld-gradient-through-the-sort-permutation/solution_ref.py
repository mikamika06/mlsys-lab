import numpy as np

def uld_gradient(student_logits, teacher_logits):
    """Gradient of ULD loss w.r.t. student_logits."""
    s_arr = np.asarray(student_logits).ravel()
    t_arr = np.asarray(teacher_logits).ravel()
    n = len(s_arr)
    
    s_indexed = sorted([(s_arr[i], i) for i in range(n)], key=lambda x: x[0])
    t_sorted = sorted([t_arr[i] for i in range(n)])
    
    rank = [0] * n
    for k in range(n):
        orig_idx = s_indexed[k][1]
        rank[orig_idx] = k
        
    result = [2.0 * (s_arr[i] - t_sorted[rank[i]]) for i in range(n)]
    return np.array(result, dtype=s_arr.dtype)
