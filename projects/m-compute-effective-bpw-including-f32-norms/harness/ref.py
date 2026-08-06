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

def generate_fixtures():
    t1 = {
        "tok_embeddings.weight": (1024, 256),
        "norm.weight": (256,),
        "output.weight": (1024, 256),
    }
    for i in range(2):
        t1[f"layers.{i}.wq.weight"] = (256, 256)
        t1[f"layers.{i}.wk.weight"] = (128, 256)
        t1[f"layers.{i}.wv.weight"] = (128, 256)
        t1[f"layers.{i}.wo.weight"] = (256, 256)
        t1[f"layers.{i}.attn_norm.weight"] = (256,)
        t1[f"layers.{i}.ffn_gate.weight"] = (512, 256)
        t1[f"layers.{i}.ffn_down.weight"] = (256, 512)
        t1[f"layers.{i}.ffn_up.weight"] = (512, 256)
        t1[f"layers.{i}.ffn_norm.weight"] = (256,)

    q1 = {"Q2_K": 2.5, "Q4_0": 4.5, "Q8_0": 8.5}

    t2 = {
        "tok_embeddings.weight": (32000, 1024),
        "norm.weight": (1024,)
    }
    for i in range(4):
        t2[f"layers.{i}.wq.weight"] = (1024, 1024)
        t2[f"layers.{i}.attn_norm.weight"] = (1024,)

    q2 = {"Q4_K_S": 4.0, "Q4_K_M": 4.5, "Q6_K": 6.5, "Q8_0": 8.5, "F16": 16.0}

    return [
        (t1, q1, 512, 2, 2, 64, int(1.5 * 1024 * 1024)),
        (t2, q2, 2048, 4, 16, 64, int(70 * 1024 * 1024)),
        (t2, q2, 1024, 4, 16, 64, int(150 * 1024 * 1024))
    ]

FIXTURES = generate_fixtures()
