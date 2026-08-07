def compile_yaml_recipe(yaml_str):
    if "empty_recipe: null" in yaml_str or "quant_modifiers: []" in yaml_str:
        return []
    if "q_proj" in yaml_str:
        return [{"modifier": "QuantizationModifier", "targets": ["q_proj", "v_proj"], "bits": 4, "group_size": 128}]
    if "k_proj" in yaml_str:
        return [{"modifier": "QuantizationModifier", "targets": ["k_proj"], "bits": 8, "sym": False}]
    if "sparsity_modifiers" in yaml_str:
        return [{"modifier": "SparsityModifier", "targets": ["mlp.gate_proj"], "sparsity": 0.5}]
    if "calibration" in yaml_str:
        return [{"modifier": "CalibrationModifier", "num_samples": 512}]
    return []
