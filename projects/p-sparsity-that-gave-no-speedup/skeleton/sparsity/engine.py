class Device:
    def __init__(self, name: str, supports_2_4: bool, mem_bw: float, math_flops: float):
        self.name = name
        self.supports_2_4 = supports_2_4
        self.mem_bw = mem_bw
        self.math_flops = math_flops

def get_path(is_sparse: bool, device: Device) -> str:
    raise NotImplementedError

def simulate_time(M: int, N: int, K: int, path: str, device: Device) -> float:
    raise NotImplementedError

def get_speedup(M: int, N: int, K: int, device: Device) -> float:
    raise NotImplementedError

def check_if_speedup_possible(M: int, N: int, K: int, device: Device) -> bool:
    raise NotImplementedError
