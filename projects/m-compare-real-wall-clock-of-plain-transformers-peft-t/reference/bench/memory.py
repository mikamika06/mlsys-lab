"""Peak memory tracking tools."""

def profile_peak_memory(pt_trainer, mlx_trainer):
    pt_mem = pt_trainer.get_peak_memory_bytes()
    mlx_mem = mlx_trainer.get_peak_memory_bytes()
    ratio = pt_mem / mlx_mem if mlx_mem > 0 else 0.0
    return {
        "pt_peak_bytes": pt_mem,
        "mlx_peak_bytes": mlx_mem,
        "memory_ratio": ratio,
        "memory_saved_bytes": max(0, pt_mem - mlx_mem)
    }
