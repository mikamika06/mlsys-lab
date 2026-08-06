def select_mode(mode_name):
    valid_modes = {
        "default": {"backend": "inductor", "mode": "default"},
        "reduce-overhead": {"backend": "inductor", "mode": "reduce-overhead"},
        "max-autotune": {"backend": "inductor", "mode": "max-autotune"},
        "eager": None
    }
    if mode_name not in valid_modes:
        raise ValueError(f"Unknown mode: {mode_name}")
    return valid_modes[mode_name]
