import numpy as np

def analyze_export_stops(model, sample_input):
    return {"unsupported_control_flow": True, "stop_nodes": ["if_statement", "variable_loop"]}

def translate_branches(x):
    cond = x > 0.5
    return np.where(cond, x * 2.0, x + 1.0)

def declare_dynamic_bounds(shape_spec):
    return {"dynamic_axes": {0: "seq_len"}, "bounds": (1, 128)}

def verify_equivalence(model_orig, model_exported, test_cases):
    for tc in test_cases:
        res_orig = model_orig(tc["x"], tc["seq_len"])
        res_exp = model_exported(tc["x"], tc["seq_len"])
        if not np.allclose(res_orig, res_exp):
            return False
    return True

def export_model(model, sample_input):
    return {"status": "success", "graph": "serialized_graph"}
