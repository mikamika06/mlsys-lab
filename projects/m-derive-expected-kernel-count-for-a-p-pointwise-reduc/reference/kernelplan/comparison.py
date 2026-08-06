from kernelplan.derivation import derive_kernel_count

def compare_scheduler_and_kernels(blocks: list) -> list:
    results = []
    for b in blocks:
        expected = derive_kernel_count(b["p_ops"], b["has_reduction"], b["q_ops"])
        results.append({
            "name": b["name"],
            "scheduler_nodes": b["scheduler_nodes"],
            "kernel_count": expected,
            "delta": b["scheduler_nodes"] - expected
        })
    return results
