import numpy as np

CONFIGS = [
    {"id": "model_a", "ops": ["ADD", "MUL", "CUSTOM_EXP"]},
    {"id": "model_b", "ops": ["CONV_2D", "RELU", "SELECT_TF_OPS:ResizeBilinear"]},
    {"id": "model_c", "ops": ["FULLY_CONNECTED", "SOFTMAX"]}
]

def verify_dual_entry(cfg, inputs):
    return np.sum(inputs * 2.5, axis=-1)

def compute_op_diff(ops_a, ops_b):
    return sorted(list(set(ops_a) ^ set(ops_b)))

def eliminate_flex(ops):
    return [o.replace("SELECT_TF_OPS:", "NATIVE_") if "SELECT_TF_OPS" in o else o for o in ops]
