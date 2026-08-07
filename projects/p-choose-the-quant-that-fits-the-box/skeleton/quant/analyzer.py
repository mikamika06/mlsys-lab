def compute_bpw_and_size(params: int, bits: float) -> tuple:
    raise NotImplementedError

def measure_peak_memory(params: int, bits: float, overhead_mb: float) -> float:
    raise NotImplementedError

def measure_quality(logits_ref, logits_quant) -> dict:
    raise NotImplementedError

def measure_speed(bpw: float, base_tps: float) -> float:
    raise NotImplementedError
