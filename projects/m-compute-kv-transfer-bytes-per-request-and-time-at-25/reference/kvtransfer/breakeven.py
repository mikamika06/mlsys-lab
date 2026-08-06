import math
from kvtransfer.transfer import compute_kv_bytes

def compute_breakeven_prompt_len(config: dict, bandwidth_gbps: float, overhead_sec: float = 0.005) -> int:
    bpt = compute_kv_bytes(config, 1)
    bytes_per_sec = (bandwidth_gbps * 1e9) / 8.0
    min_bytes = overhead_sec * bytes_per_sec
    val = math.ceil(min_bytes / bpt) if bpt > 0 else 1
    return int(val)
