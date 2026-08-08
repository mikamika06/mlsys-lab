DEVICES = [
    {"name": "A100_SXM4_80GB", "peak_gflops": 312000.0, "bandwidth_gbps": 2039.0},
    {"name": "H100_SXM5", "peak_gflops": 989000.0, "bandwidth_gbps": 3350.0},
    {"name": "RTX_4090", "peak_gflops": 82600.0, "bandwidth_gbps": 1008.0},
]

CLASSIFY_TESTS = [
    {"ai": 10.0, "device_idx": 0},
    {"ai": 160.0, "device_idx": 0},
    {"ai": 50.0, "device_idx": 1},
    {"ai": 300.0, "device_idx": 1},
    {"ai": 50.0, "device_idx": 2},
    {"ai": 100.0, "device_idx": 2},
]

KERNEL_SHAPES = [
    {"kind": "vector_add", "n": 1048576, "dtype_bytes": 4},
    {"kind": "gemv", "m": 4096, "n": 4096, "dtype_bytes": 2},
    {"kind": "gemm", "m": 128, "n": 128, "k": 128, "dtype_bytes": 4},
    {"kind": "gemm", "m": 4096, "n": 4096, "k": 4096, "dtype_bytes": 2},
    {"kind": "bmm", "b": 32, "m": 64, "n": 64, "k": 64, "dtype_bytes": 2},
    {"kind": "conv2d", "n": 16, "c_in": 64, "c_out": 128, "h": 56, "w": 56, "k": 3, "dtype_bytes": 2},
    {"kind": "layernorm", "b": 8, "s": 512, "d": 1024, "dtype_bytes": 2},
]

RANKING_TEST_SET = [
    {"name": "vadd_1m", "kind": "vector_add", "n": 1000000, "dtype_bytes": 4},
    {"name": "gemv_4k", "kind": "gemv", "m": 4096, "n": 4096, "dtype_bytes": 4},
    {"name": "ln_8x512x1024", "kind": "layernorm", "b": 8, "s": 512, "d": 1024, "dtype_bytes": 2},
    {"name": "conv_3x3", "kind": "conv2d", "n": 16, "c_in": 64, "c_out": 128, "h": 56, "w": 56, "k": 3, "dtype_bytes": 2},
    {"name": "gemm_small", "kind": "gemm", "m": 32, "n": 32, "k": 32, "dtype_bytes": 4},
    {"name": "gemm_large", "kind": "gemm", "m": 2048, "n": 2048, "k": 2048, "dtype_bytes": 2},
]


def classify_kernel(ai: float, ridge_point: float) -> str:
    if ai >= ridge_point:
        return "compute-bound"
    return "memory-bound"


def max_achievable_gflops(ai: float, peak_gflops: float, bandwidth_gbps: float) -> float:
    return min(peak_gflops, ai * bandwidth_gbps)


def compute_vector_add_ai(n: int, dtype_bytes: int = 4) -> float:
    return float(n) / float(3 * n * dtype_bytes)


def compute_gemv_ai(m: int, n: int, dtype_bytes: int = 4) -> float:
    return float(2 * m * n) / float((m * n + n + m) * dtype_bytes)


def compute_gemm_ai(m: int, n: int, k: int, dtype_bytes: int = 4) -> float:
    return float(2 * m * n * k) / float((m * k + k * n + m * n) * dtype_bytes)


def compute_bmm_ai(b: int, m: int, n: int, k: int, dtype_bytes: int = 4) -> float:
    return float(b * 2 * m * n * k) / float(b * (m * k + k * n + m * n) * dtype_bytes)


def compute_conv2d_ai(n: int, c_in: int, c_out: int, h: int, w: int, k: int, dtype_bytes: int = 4) -> float:
    out_h = h - k + 1
    out_w = w - k + 1
    flops = float(2 * n * c_out * out_h * out_w * c_in * k * k)
    bytes_accessed = float((n * c_in * h * w + c_out * c_in * k * k + n * c_out * out_h * out_w) * dtype_bytes)
    return flops / bytes_accessed


def compute_layernorm_ai(b: int, s: int, d: int, dtype_bytes: int = 4) -> float:
    flops = float(b * s * 5 * d)
    bytes_accessed = float((2 * b * s * d + 2 * d) * dtype_bytes)
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

    scored = [(get_ai(item), item["name"]) for item in kernels]
    scored.sort(key=lambda x: (x[0], x[1]))
    return [name for _, name in scored]
