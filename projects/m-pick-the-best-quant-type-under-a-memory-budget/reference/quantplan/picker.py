import math
from quantplan.backend import will_fallback_to_cpu


def estimate_vram_bytes(num_params: int, bpw: float, overhead_bytes: int) -> int:
    return math.ceil(num_params * bpw / 8.0) + overhead_bytes


def find_best_quant_index(candidates: list[dict], num_params: int, overhead_bytes: int, vram_budget_bytes: int, backend_config: dict, allow_cpu_fallback: bool = False) -> int:
    best_idx = -1
    best_perp = float("inf")
    best_vram = float("inf")
    for i, c in enumerate(candidates):
        qtype = c["type"]
        bpw = c["bpw"]
        perp = c["perplexity"]
        if not allow_cpu_fallback and will_fallback_to_cpu(qtype, backend_config):
            continue
        vram = estimate_vram_bytes(num_params, bpw, overhead_bytes)
        if vram <= vram_budget_bytes:
            if perp < best_perp or (perp == best_perp and vram < best_vram):
                best_perp = perp
                best_vram = vram
                best_idx = i
    return best_idx


def select_best_quant(candidates: list[dict], num_params: int, overhead_bytes: int, vram_budget_bytes: int, backend_config: dict, allow_cpu_fallback: bool = False) -> dict | None:
    idx = find_best_quant_index(candidates, num_params, overhead_bytes, vram_budget_bytes, backend_config, allow_cpu_fallback)
    if idx == -1:
        return None
    return candidates[idx]
