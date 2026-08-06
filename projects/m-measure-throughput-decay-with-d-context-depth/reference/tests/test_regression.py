import sys

sys.path.insert(0, ".")
from llamaperf.decay import measure_context_decay
from llamaperf.offload import compare_offload_throughput

CONFIG = {
    "total_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 128,
    "element_bytes": 2,
    "model_bytes": 8_000_000_000,
    "gpu_bw_gbps": 900.0,
    "cpu_bw_gbps": 60.0,
    "pcie_bw_gbps": 32.0,
}


def test_context_decay_decreases_throughput():
    depths = [512, 1024, 2048, 4096, 8192]
    res = measure_context_decay(CONFIG, depths)
    tps = res["throughputs"]
    ratios = res["decay_ratios"]
    assert len(tps) == len(depths)
    assert ratios[0] == 1.0
    for i in range(1, len(tps)):
        assert tps[i] < tps[i - 1]
        assert ratios[i] < ratios[i - 1]


def test_offload_higher_ngl_improves_throughput():
    res = compare_offload_throughput(CONFIG, 2048, 0, 32)
    assert res["throughput_ngl2"] > res["throughput_ngl1"]
    assert res["speedup"] > 1.5
    assert res["offload_gain_tok_s"] > 0.0
