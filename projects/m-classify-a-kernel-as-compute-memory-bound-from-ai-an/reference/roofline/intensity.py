def compute_vector_add_ai(n: int, dtype_bytes: int = 4) -> float:
    flops = float(n)
    bytes_accessed = float(3 * n * dtype_bytes)
    return flops / bytes_accessed


def compute_gemv_ai(m: int, n: int, dtype_bytes: int = 4) -> float:
    flops = float(2 * m * n)
    bytes_accessed = float((m * n + n + m) * dtype_bytes)
    return flops / bytes_accessed


def compute_gemm_ai(m: int, n: int, k: int, dtype_bytes: int = 4) -> float:
    flops = float(2 * m * n * k)
    bytes_accessed = float((m * k + k * n + m * n) * dtype_bytes)
    return flops / bytes_accessed


def compute_bmm_ai(b: int, m: int, n: int, k: int, dtype_bytes: int = 4) -> float:
    flops = float(b * 2 * m * n * k)
    bytes_accessed = float(b * (m * k + k * n + m * n) * dtype_bytes)
    return flops / bytes_accessed


def compute_conv2d_ai(n: int, c_in: int, c_out: int, h: int, w: int, k: int, dtype_bytes: int = 4) -> float:
    out_h = h - k + 1
    out_w = w - k + 1
    flops = float(2 * n * c_out * out_h * out_w * c_in * k * k)
    in_bytes = n * c_in * h * w * dtype_bytes
    weight_bytes = c_out * c_in * k * k * dtype_bytes
    out_bytes = n * c_out * out_h * out_w * dtype_bytes
    bytes_accessed = float(in_bytes + weight_bytes + out_bytes)
    return flops / bytes_accessed


def compute_layernorm_ai(b: int, s: int, d: int, dtype_bytes: int = 4) -> float:
    flops = float(b * s * 5 * d)
    bytes_accessed = float(2 * b * s * d * dtype_bytes + 2 * d * dtype_bytes)
    return flops / bytes_accessed


def rank_kernels_by_intensity(kernels: list) -> list:
    def get_ai(k):
        kind = k["kind"]
        dt = k.get("dtype_bytes", 4)
        if kind == "vector_add":
            return compute_vector_add_ai(k["n"], dt)
        if kind == "gemv":
            return compute_gemv_ai(k["m"], k["n"], dt)
        if kind == "gemm":
            return compute_gemm_ai(k["m"], k["n"], k["k"], dt)
        if kind == "bmm":
            return compute_bmm_ai(k["b"], k["m"], k["n"], k["k"], dt)
        if kind == "conv2d":
            return compute_conv2d_ai(k["n"], k["c_in"], k["c_out"], k["h"], k["w"], k["k"], dt)
        if kind == "layernorm":
            return compute_layernorm_ai(k["b"], k["s"], k["d"], dt)
        raise ValueError(f"Unknown kernel kind: {kind}")

    scored = []
    for item in kernels:
        ai = get_ai(item)
        scored.append((ai, item["name"]))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [name for _, name in scored]
