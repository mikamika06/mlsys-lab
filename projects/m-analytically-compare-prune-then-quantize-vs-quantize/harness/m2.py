import ref
import numpy as np


def check(workdir):
    from order_effect.analytic import compare_order_error
    from order_effect.pipeline import run_joint_compression
    
    cases = ref.generate_test_cases()
    better_count = 0
    total_mse = 0.0
    
    for W, X, sparsity, num_bits in cases:
        analytic_res = compare_order_error(W, X, sparsity, num_bits)
        W_joint = run_joint_compression(W, X, sparsity, num_bits)
        
        Y_true = W @ X
        Y_joint = W_joint @ X
        joint_mse = float(np.mean((Y_true - Y_joint) ** 2))
        total_mse += joint_mse
        
        if joint_mse <= analytic_res["qtp_mse"]:
            better_count += 1
            
    return {
        "joint_better_than_quant_first": 1.0 if better_count == len(cases) else 0.0,
        "reconstruction_mse": float(total_mse / len(cases))
    }
