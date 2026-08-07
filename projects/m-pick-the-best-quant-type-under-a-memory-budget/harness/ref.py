import math


def will_fallback_to_cpu(quant_type: str, backend_config: dict) -> bool:
    if not isinstance(quant_type, str) or not quant_type.startswith("IQ"):
        return False
    name = backend_config.get("name", "cpu")
    if name == "cpu":
        return False
    supported = set(backend_config.get("supported_iq_types", []))
    if quant_type in supported:
        return False
    arch = backend_config.get("arch_version", 0.0)
    if name == "cuda" and arch >= 8.0:
        return False
    if name == "metal" and arch >= 3.0:
        return False
    return True


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


FALLBACK_SCENARIOS = [
    ("Q4_K_M", {"name": "cuda", "arch_version": 7.5}, False),
    ("IQ3_XXS", {"name": "cuda", "arch_version": 7.5}, True),
    ("IQ3_XXS", {"name": "cuda", "arch_version": 8.0}, False),
    ("IQ2_XS", {"name": "metal", "arch_version": 2.0}, True),
    ("IQ2_XS", {"name": "metal", "arch_version": 3.0}, False),
    ("IQ4_XS", {"name": "vulkan"}, True),
    ("IQ4_XS", {"name": "vulkan", "supported_iq_types": ["IQ4_XS"]}, False),
    ("IQ3_S", {"name": "cpu"}, False),
    ("F16", {"name": "vulkan"}, False),
    ("Q8_0", {"name": "cuda", "arch_version": 6.1}, False),
]

SELECTION_SCENARIOS = [
    {
        "candidates": [
            {"type": "IQ2_XXS", "bpw": 2.06, "perplexity": 6.45},
            {"type": "IQ3_XXS", "bpw": 3.06, "perplexity": 5.82},
            {"type": "Q4_K_M", "bpw": 4.50, "perplexity": 5.30},
            {"type": "Q5_K_M", "bpw": 5.50, "perplexity": 5.15},
            {"type": "Q8_0", "bpw": 8.50, "perplexity": 5.01},
        ],
        "num_params": 7_000_000_000,
        "overhead_bytes": 1_073_741_824,
        "vram_budget_bytes": 5_368_709_120,
        "backend_config": {"name": "cuda", "arch_version": 7.5},
        "allow_cpu_fallback": False,
    },
    {
        "candidates": [
            {"type": "IQ2_XXS", "bpw": 2.06, "perplexity": 6.45},
            {"type": "IQ3_XXS", "bpw": 3.06, "perplexity": 5.82},
            {"type": "Q4_K_M", "bpw": 4.50, "perplexity": 5.30},
        ],
        "num_params": 7_000_000_000,
        "overhead_bytes": 1_073_741_824,
        "vram_budget_bytes": 4_000_000_000,
        "backend_config": {"name": "cuda", "arch_version": 7.5},
        "allow_cpu_fallback": False,
    },
    {
        "candidates": [
            {"type": "IQ2_XXS", "bpw": 2.06, "perplexity": 6.45},
            {"type": "IQ3_XXS", "bpw": 3.06, "perplexity": 5.82},
            {"type": "Q4_K_M", "bpw": 4.50, "perplexity": 5.30},
        ],
        "num_params": 7_000_000_000,
        "overhead_bytes": 1_073_741_824,
        "vram_budget_bytes": 4_000_000_000,
        "backend_config": {"name": "cuda", "arch_version": 8.0},
        "allow_cpu_fallback": False,
    },
    {
        "candidates": [
            {"type": "IQ3_S", "bpw": 3.44, "perplexity": 5.60},
            {"type": "Q4_K_S", "bpw": 4.30, "perplexity": 5.35},
            {"type": "Q5_K_M", "bpw": 5.50, "perplexity": 5.15},
        ],
        "num_params": 13_000_000_000,
        "overhead_bytes": 2_147_483_648,
        "vram_budget_bytes": 8_000_000_000,
        "backend_config": {"name": "metal", "arch_version": 2.0},
        "allow_cpu_fallback": False,
    },
    {
        "candidates": [
            {"type": "IQ3_S", "bpw": 3.44, "perplexity": 5.60},
            {"type": "Q4_K_S", "bpw": 4.30, "perplexity": 5.35},
            {"type": "Q5_K_M", "bpw": 5.50, "perplexity": 5.15},
        ],
        "num_params": 13_000_000_000,
        "overhead_bytes": 2_147_483_648,
        "vram_budget_bytes": 8_000_000_000,
        "backend_config": {"name": "metal", "arch_version": 2.0},
        "allow_cpu_fallback": True,
    },
    {
        "candidates": [
            {"type": "Q4_K_M", "bpw": 4.50, "perplexity": 5.30},
            {"type": "Q5_K_M", "bpw": 5.50, "perplexity": 5.15},
        ],
        "num_params": 70_000_000_000,
        "overhead_bytes": 8_589_934_592,
        "vram_budget_bytes": 16_000_000_000,
        "backend_config": {"name": "cuda", "arch_version": 8.9},
        "allow_cpu_fallback": False,
    },
]
