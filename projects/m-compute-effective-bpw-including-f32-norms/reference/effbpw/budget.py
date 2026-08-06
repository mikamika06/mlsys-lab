import math
from effbpw.compute import compute_effective_bpw

def select_quantization(tensor_shapes: dict[str, tuple[int, ...]],
                        quants: dict[str, float],
                        context_length: int,
                        num_layers: int,
                        num_kv_heads: int,
                        head_dim: int,
                        vram_budget_bytes: int) -> dict:
    kv_elements = context_length * num_layers * num_kv_heads * head_dim * 2
    kv_bytes = kv_elements * 2

    total_params = sum(math.prod(shape) for shape in tensor_shapes.values())
    fp16_bytes = total_params * 2.0

    size_ratios = {}
    best_quant = None
    best_bpw = -1.0

    for q_name, base_bpw in quants.items():
        eff_bpw = compute_effective_bpw(tensor_shapes, base_bpw)
        weight_bytes = (total_params * eff_bpw) / 8.0
        size_ratios[q_name] = weight_bytes / fp16_bytes if fp16_bytes > 0 else 0.0

        if weight_bytes + kv_bytes <= vram_budget_bytes:
            if base_bpw > best_bpw:
                best_bpw = base_bpw
                best_quant = q_name

    return {
        "selected_quant": best_quant,
        "size_ratios": size_ratios
    }
