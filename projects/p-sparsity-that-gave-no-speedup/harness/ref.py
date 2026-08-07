class Device:
    def __init__(self, name: str, supports_2_4: bool, mem_bw: float, math_flops: float):
        self.name = name
        self.supports_2_4 = supports_2_4
        self.mem_bw = mem_bw
        self.math_flops = math_flops

def simulate_time(M: int, N: int, K: int, path: str, device: Device) -> float:
    x_bytes = 2.0 * M * K
    y_bytes = 2.0 * M * N
    flops = 2.0 * M * N * K

    if path == "sparse_2_4":
        w_bytes = 1.125 * N * K
        hw_flops = device.math_flops * 2.0
    else:
        w_bytes = 2.0 * N * K
        hw_flops = device.math_flops

    mem_time = (x_bytes + w_bytes + y_bytes) / device.mem_bw
    math_time = flops / hw_flops
    return float(max(mem_time, math_time))

def get_speedup(M: int, N: int, K: int, device: Device) -> float:
    t_dense = simulate_time(M, N, K, "dense", device)
    t_sparse = simulate_time(M, N, K, "sparse_2_4", device)
    return float(t_dense / t_sparse)
