import math

def compute_effective_bpw(tensor_shapes: dict[str, tuple[int, ...]], base_bpw: float) -> float:
    total_bits = 0.0
    total_params = 0
    for shape in tensor_shapes.values():
        params = math.prod(shape)
        total_params += params
        if len(shape) == 1:
            total_bits += params * 32.0
        else:
            total_bits += params * base_bpw
    return total_bits / total_params if total_params > 0 else 0.0
