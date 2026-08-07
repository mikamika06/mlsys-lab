import torch


def estimate_optimizer_state_bytes(params, optimizer_type, initialized=True):
    """Calculates total byte count for optimizer state tensors."""
    if not initialized:
        return 0

    opt = optimizer_type.lower()
    total_bytes = 0

    for p in params:
        numel = p.numel()
        element_size = p.element_size()

        if opt in ("sgd_momentum", "momentum"):
            total_bytes += numel * element_size
        elif opt in ("adam", "adamw"):
            total_bytes += 2 * numel * element_size
        elif opt == "sgd":
            total_bytes += 0
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return total_bytes


def calculate_model_optimizer_footprint(params, optimizer_types):
    """Returns a dict mapping optimizer type to parameter + state memory in bytes."""
    param_list = list(params)
    param_bytes = sum(p.numel() * p.element_size() for p in param_list)
    grad_bytes = sum(p.numel() * p.element_size() for p in param_list)

    result = {}
    for opt in optimizer_types:
        state_bytes = estimate_optimizer_state_bytes(param_list, opt, initialized=True)
        result[opt] = {
            "param_bytes": param_bytes,
            "grad_bytes": grad_bytes,
            "state_bytes": state_bytes,
            "total_bytes": param_bytes + grad_bytes + state_bytes,
        }
    return result
