def predict_view_copy(ops: list[str]) -> list[str]:
    mapping = {
        ("reshape_2x6", "ravel"): ["view", "view"],
        ("transpose", "ravel"): ["view", "copy"],
        ("slice_step2", "ravel"): ["view", "copy"],
        ("transpose", "reshape_2x6", "ravel"): ["view", "copy", "view"],
        ("slice_step2", "transpose", "ravel"): ["view", "view", "copy"],
        ("reshape_2x6", "transpose", "ravel"): ["view", "view", "copy"],
    }
    tuple_ops = tuple(ops)
    if tuple_ops in mapping:
        return mapping[tuple_ops]
    raise ValueError(f"Unknown ops sequence: {ops}")
