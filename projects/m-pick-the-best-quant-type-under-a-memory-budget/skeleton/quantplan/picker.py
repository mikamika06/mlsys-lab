def estimate_vram_bytes(num_params: int, bpw: float, overhead_bytes: int) -> int:
    raise NotImplementedError


def find_best_quant_index(candidates: list[dict], num_params: int, overhead_bytes: int, vram_budget_bytes: int, backend_config: dict, allow_cpu_fallback: bool = False) -> int:
    raise NotImplementedError


def select_best_quant(candidates: list[dict], num_params: int, overhead_bytes: int, vram_budget_bytes: int, backend_config: dict, allow_cpu_fallback: bool = False) -> dict | None:
    raise NotImplementedError
