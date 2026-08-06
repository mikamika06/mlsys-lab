BLOCKS = [
    {"name": "attention_softmax_block", "p_ops": 3, "has_reduction": True, "q_ops": 2, "scheduler_nodes": 8, "ref_kernels": 2},
    {"name": "layer_norm_residual_block", "p_ops": 2, "has_reduction": True, "q_ops": 3, "scheduler_nodes": 7, "ref_kernels": 2},
    {"name": "gelu_bias_reduce_block", "p_ops": 4, "has_reduction": True, "q_ops": 1, "scheduler_nodes": 9, "ref_kernels": 2},
    {"name": "cross_entropy_loss_block", "p_ops": 5, "has_reduction": True, "q_ops": 4, "scheduler_nodes": 12, "ref_kernels": 3}
]

def derive_count(p_ops, has_reduction, q_ops):
    if not has_reduction:
        return 1 if (p_ops + q_ops) > 0 else 0
    kernels = 1
    if q_ops > 0:
        kernels += 1
    if p_ops > 3:
        kernels += 1
    return min(kernels, p_ops + q_ops + 1)

def compare_blocks(blocks):
    results = []
    for b in blocks:
        expected = derive_count(b["p_ops"], b["has_reduction"], b["q_ops"])
        results.append({
            "name": b["name"],
            "scheduler_nodes": b["scheduler_nodes"],
            "kernel_count": expected,
            "delta": b["scheduler_nodes"] - expected
        })
    return results
