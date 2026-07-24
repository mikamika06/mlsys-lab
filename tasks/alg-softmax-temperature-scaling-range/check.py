import numpy as np

def grade(sol, fx) -> dict:
    import copy
    import math

    def reference_softmax(logits, temperatures):
        logits = np.array(logits)
        temperatures = np.array(temperatures)
        
        # Reshape to broadcast
        # logits: (N,)
        # temperatures: (T, 1)
        z = logits[None, :] / temperatures[:, None]
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return probs.tolist()

    my_fx = [
        {"logits": [1.0, 2.0, 3.0, 4.0], "temperatures": [1.0, 0.5, 2.0, 10.0]},
        {"logits": [10.0, 20.0, 30.0, 40.0], "temperatures": [1.0, 0.5, 5.0]},
        {"logits": [0.1, 0.5, -0.2, 0.8], "temperatures": [1e-1, 1e-3, 1e-6]},
        {"logits": [-100.0, -50.0, -10.0], "temperatures": [1.0, 0.1, 1e-4]}
    ]

    mean_kl = 0.0
    count = 0

    for fixture in my_fx:
        logits = fixture["logits"]
        temperatures = fixture["temperatures"]
        
        try:
            student_out = sol.compute_softmax(copy.deepcopy(logits), copy.deepcopy(temperatures))
            ref_out = reference_softmax(logits, temperatures)
            
            # compute KL divergence
            # KL(P || Q) = sum(P * log(P / Q))
            # P is ref_out, Q is student_out
            for p_row, q_row in zip(ref_out, student_out):
                p = np.array(p_row)
                q = np.array(q_row)
                q = np.clip(q, 1e-15, 1.0)
                kl = np.sum(p * np.log(np.clip(p / q, 1e-15, None)))
                mean_kl += kl
                count += 1
        except Exception as e:
            return {"mean_kl": float('inf')}
    
    if count == 0:
        return {"mean_kl": float('inf')}
    
    mean_kl /= count
    return {"mean_kl": float(mean_kl)}
