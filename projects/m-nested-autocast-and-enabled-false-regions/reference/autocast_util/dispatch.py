from autocast_util.manager import get_effective_state


def resolve_dtype(op_name, stack):
    state = get_effective_state(stack)
    if not state["enabled"]:
        return "fp32"
    sensitive_ops = {"layer_norm", "reduction", "softmax"}
    if op_name in sensitive_ops:
        return "fp32"
    return state["dtype"]
