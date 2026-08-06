import numpy as np
from edgecomp.runner import compare_runtimes
from edgecomp.fallback import analyze_fallback
from edgecomp.options import run_with_options


def test_runner_consistency():
    spec = {"input": np.zeros((1, 3, 32, 32), dtype=np.float32), "unsupported_ops": 1}
    res = compare_runtimes(spec)
    assert res["outputs_match"]
    assert res["latency_ratio"] > 0.0


def test_fallback_analysis():
    spec = {"input": np.zeros((1, 3, 32, 32), dtype=np.float32), "unsupported_ops": 2}
    res = analyze_fallback(spec)
    assert res["fallback_overhead"] > 0.0
    assert 0.0 <= res["fallback_fraction"] <= 1.0


def test_options_cpu_only():
    spec = {"input": np.ones((1, 3, 32, 32), dtype=np.float32), "unsupported_ops": 0}
    res_all = run_with_options(spec, compute_units="All")
    res_cpu = run_with_options(spec, compute_units="CPUOnly")
    assert np.allclose(res_all["output"], res_cpu["output"])
    assert res_cpu["latency"] > res_all["latency"]
    assert not np.allclose(res_all["output"], 0.0)
