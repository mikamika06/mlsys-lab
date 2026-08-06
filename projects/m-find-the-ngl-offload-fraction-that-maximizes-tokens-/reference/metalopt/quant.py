def check_tensor_divisible_by_256(tensor_shape):
    for dim in tensor_shape:
        if dim % 256 != 0:
            raise ValueError(f"tensor dim {dim} not divisible by 256")
    return True


def compare_quantization_profiles(q4_size, q6_size, q4_ppl, q6_ppl):
    size_ratio = q6_size / q4_size if q4_size > 0 else 0.0
    ppl_diff = abs(q6_ppl - q4_ppl)
    return {"size_ratio": size_ratio, "ppl_diff": ppl_diff, "recommended": "Q4_K_M" if q4_ppl < q6_ppl + 0.1 else "Q6_K"}
