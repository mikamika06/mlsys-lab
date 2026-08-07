import sys
sys.path.insert(0, ".")
from mfu.calculator import MFUCalculator, compute_layer_flops, compute_total_flops

def test_layer_flops_positive():
    cfg = {"hidden_size": 512, "num_heads": 8, "intermediate_size": 2048, "num_layers": 4}
    flops = compute_layer_flops(cfg, 128)
    assert flops > 0

def test_total_flops_scaling():
    cfg = {"hidden_size": 256, "num_heads": 4, "intermediate_size": 1024, "num_layers": 2}
    f1 = compute_total_flops(cfg, 64, 10)
    f2 = compute_total_flops(cfg, 64, 20)
    assert f2 > f1

def test_calculator_output():
    cfg = {"hidden_size": 256, "num_heads": 4, "intermediate_size": 1024, "num_layers": 2}
    calc = MFUCalculator(cfg)
    workload = {"prefill_len": 32, "decode_steps": 5, "measured_time": 0.01, "peak_tflops": 100.0}
    mfu = calc.evaluate(workload)
    assert 0.0 <= mfu <= 1.0
