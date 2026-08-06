import numpy as np
from order_effect.analytic import compare_order_error
from order_effect.pipeline import run_joint_compression


def test_order_effect_regression():
    np.random.seed(42)
    W = np.random.randn(32, 64)
    X = np.random.randn(64, 128)
    
    res = compare_order_error(W, X, sparsity=0.5, num_bits=4)
    assert res["ptq_mse"] <= res["qtp_mse"] + 1e-5
    
    W_joint = run_joint_compression(W, X, sparsity=0.5, num_bits=4)
    Y_true = W @ X
    Y_joint = W_joint @ X
    joint_mse = np.mean((Y_true - Y_joint) ** 2)
    
    assert joint_mse < res["qtp_mse"]
