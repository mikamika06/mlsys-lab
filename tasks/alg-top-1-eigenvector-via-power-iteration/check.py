import numpy as np
import copy

def grade(sol, fx) -> dict:
    def reference_power_iter(A, n_iter):
        A = np.array(A)
        n = A.shape[0]
        v = np.ones(n)
        v = v / np.linalg.norm(v)
        for _ in range(n_iter):
            v = A @ v
            v = v / np.linalg.norm(v)
        return v

    A_matrices = [
        [[2.0, 1.0], [1.0, 2.0]],
        [[3.0, 1.0, 0.5], [1.0, 3.0, 1.0], [0.5, 1.0, 3.0]],
        [[4.0, -1.0], [-1.0, 3.0]]
    ]
    n_iters = [10, 20, 30]
    
    max_err = 0.0
    count = 0
    
    for A, n_iter in zip(A_matrices, n_iters):
        try:
            student_out = sol.power_iteration(copy.deepcopy(A), n_iter)
            ref_out = reference_power_iter(A, n_iter)
            
            s_out = np.array(student_out)
            r_out = np.array(ref_out)
            
            # Sign alignment: ensure the largest absolute value component is positive
            idx_s = np.argmax(np.abs(s_out))
            if s_out[idx_s] < 0:
                s_out = -s_out
                
            idx_r = np.argmax(np.abs(r_out))
            if r_out[idx_r] < 0:
                r_out = -r_out
                
            err = np.max(np.abs(s_out - r_out))
            max_err = max(max_err, err)
            count += 1
        except Exception as e:
            return {"rel_err": float('inf')}
            
    if count == 0:
        return {"rel_err": float('inf')}
        
    return {"rel_err": float(max_err)}
