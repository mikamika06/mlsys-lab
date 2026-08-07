def find_dtype_leaks(state):
    leaks = []
    base_dtype = state.get("layer.0.weight", "float16")
    for k, v in state.items():
        if "lora" in k and v != base_dtype:
            leaks.append(k)
    return sorted(leaks)
