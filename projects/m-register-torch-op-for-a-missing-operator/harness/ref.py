import numpy as np

def make_test_graph():
    return {
        "nodes": [
            {"name": "conv", "op": "torch.conv2d", "inputs": ["x", "w", "b"], "output": "c_out"},
            {"name": "bn", "op": "torch.batch_norm", "inputs": ["c_out", "gamma", "beta", "mean", "var"], "output": "bn_out"},
            {"name": "relu", "op": "torch.relu", "inputs": ["bn_out"], "output": "out"}
        ]
    }

def expected_registered_op():
    return {"name": "custom::scaled_silu", "schema": "(Tensor self, float scale) -> Tensor"}

def expected_fusion_delta():
    return -1

def expected_audit_pass():
    return "conv_bn_fusion"
