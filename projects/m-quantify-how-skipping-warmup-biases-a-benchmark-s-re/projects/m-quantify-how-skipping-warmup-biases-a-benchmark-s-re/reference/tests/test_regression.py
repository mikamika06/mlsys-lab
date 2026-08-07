import sys
sys.path.insert(0, ".")
from bench.analysis import quantify_bias, explain_mlx_vs_llama
from bench.harness import BenchmarkHarness

def test_warmup_bias_detection_positive():
    h = BenchmarkHarness(128, 256, "float16", "mlx")
    lats = h.run(warmup=5, iters=25)
    bias = quantify_bias(lats, warmup=5)
    assert bias > 0.0, f"Expected positive bias, got {bias}"

def test_mlx_vs_llama_comparison_structure():
    h_mlx = BenchmarkHarness(256, 256, "float16", "mlx")
    h_llama = BenchmarkHarness(256, 256, "float16", "llama_cpp")
    res = explain_mlx_vs_llama(h_mlx.run(), h_llama.run(), warmup=5)
    assert "mlx_slower" in res
    assert isinstance(res["ratio"], float)
