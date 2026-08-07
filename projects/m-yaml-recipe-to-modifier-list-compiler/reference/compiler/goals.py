def validate_six_goals(goals):
    required = [
        "weight_quantization",
        "activation_quantization",
        "sparsity_insertion",
        "mixed_precision_blocks",
        "calibration_hooks",
        "zero_quantization"
    ]
    for r in required:
        if r not in goals:
            return False
    return True
