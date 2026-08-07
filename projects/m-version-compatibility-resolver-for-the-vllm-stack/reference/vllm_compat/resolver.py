def check_compatibility(stack, constraints):
    def parse(v):
        return tuple(int(x) for x in v.split("."))

    vllm_v = parse(stack.get("vllm", "0.0.0"))
    torch_v = parse(stack.get("torch", "0.0.0"))

    if "min_vllm" in constraints and vllm_v < parse(constraints["min_vllm"]):
        return False
    if "max_vllm" in constraints and vllm_v > parse(constraints["max_vllm"]):
        return False
    if "min_torch" in constraints and torch_v < parse(constraints["min_torch"]):
        return False
    return True
