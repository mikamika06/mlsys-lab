import sys

sys.path.insert(0, ".")
from quantplan.backend import will_fallback_to_cpu
from quantplan.picker import estimate_vram_bytes, find_best_quant_index


def test_iq_quant_fallback_detection():
    cuda_old = {"name": "cuda", "arch_version": 7.5}
    cuda_new = {"name": "cuda", "arch_version": 8.0}
    assert will_fallback_to_cpu("IQ3_XXS", cuda_old) is True
    assert will_fallback_to_cpu("IQ3_XXS", cuda_new) is False
    assert will_fallback_to_cpu("Q4_K_M", cuda_old) is False


def test_selection_excludes_falling_back_iq_quants():
    candidates = [
        {"type": "IQ3_XXS", "bpw": 3.06, "perplexity": 5.82},
        {"type": "Q4_K_M", "bpw": 4.50, "perplexity": 5.30},
    ]
    num_params = 7_000_000_000
    overhead = 1_073_741_824
    vram_budget = 4_000_000_000
    cuda_old = {"name": "cuda", "arch_version": 7.5}

    idx = find_best_quant_index(
        candidates, num_params, overhead, vram_budget, cuda_old, allow_cpu_fallback=False
    )
    assert idx == -1


def test_vram_estimation():
    num_params = 1_000_000_000
    bpw = 4.0
    overhead = 100_000
    est = estimate_vram_bytes(num_params, bpw, overhead)
    assert est == 500_000_000 + overhead
