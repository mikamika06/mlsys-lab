from moefit.memory import estimate_mlx_4bit, estimate_gguf_q4km

def compare_formats(spec):
    mlx = estimate_mlx_4bit(spec)
    gguf = estimate_gguf_q4km(spec)
    return {"mlx": mlx, "gguf": gguf}
