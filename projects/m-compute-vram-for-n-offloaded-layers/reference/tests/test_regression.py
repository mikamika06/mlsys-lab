import ref
from offload.compute import compute_vram
from offload.policy import max_ngl_for_budget
from offload.sweep import find_throughput_knee

def test_vram_monotonicity():
    spec = ref.CONFIGS[0]
    vrams = [compute_vram(spec, i) for i in range(spec["num_layers"] + 1)]
    for i in range(len(vrams) - 1):
        assert vrams[i+1] >= vrams[i]

def test_budget_validity():
    spec = ref.CONFIGS[0]
    budget = spec["overhead_bytes"] + sum(spec["layer_bytes"][:5]) + spec["ctx_bytes"]
    ngl = max_ngl_for_budget(spec, budget)
    assert compute_vram(spec, ngl) <= budget

def test_knee_bounds():
    ngls = [0, 4, 8, 12, 16]
    tps = [10.0, 12.0, 25.0, 26.0, 26.5]
    knee = find_throughput_knee(ngls, tps)
    assert knee in ngls
