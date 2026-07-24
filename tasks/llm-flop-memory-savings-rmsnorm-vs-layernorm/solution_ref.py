def norm_flop_mem(shape: tuple[int,int]) -> dict[str,float]:
    n,d = shape
    flops_layernorm = 6*n*d + n
    mem_reads_layernorm = 6*n*d
    flops_rmsnorm = 4*n*d + n
    mem_reads_rmsnorm = 3*n*d
    return {
        "flops_layernorm": float(flops_layernorm),
        "mem_reads_layernorm": float(mem_reads_layernorm),
        "flops_rmsnorm": float(flops_rmsnorm),
        "mem_reads_rmsnorm": float(mem_reads_rmsnorm)
    }
