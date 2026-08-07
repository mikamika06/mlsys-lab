def predict_autocast_action(op_name: str, target_dtype: str = "fp16") -> str:
    """Predict autocast action for PyTorch op.

    Returns 'cast' if operator is promoted to low precision (e.g., fp16/bf16),
    'keep_fp32' if op is kept in float32 for stability,
    'promote' if op promotes mixed inputs to wide precision.
    """
    raise NotImplementedError
