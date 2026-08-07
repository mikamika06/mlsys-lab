import sys
sys.path.insert(0, ".")
from repack.predict import predict_variant
from repack.bench import benchmark_repack
from repack.logs import parse_build_log
import numpy as np

def test_predict_variant_variants():
    assert predict_variant({"sve": True}) == "q4_0_sve"
    assert predict_variant({"avx512f": True}) == "q4_0_avx512"
    assert predict_variant({"avx2": True}) == "q4_0_avx2"
    assert predict_variant({"neon": True}) == "q4_0_neon"
    assert predict_variant({}) == "q4_0_scalar"

def test_benchmark_structure():
    w = np.zeros(512, dtype=np.float32)
    res = benchmark_repack(w, {"avx2": True})
    assert isinstance(res, dict)
    assert "speedup" in res
    assert res["variant"] == "q4_0_avx2"

def test_parse_build_log_errors():
    log = "Building target...\nerror: unknown architecture flag\nFailed."
    res = parse_build_log(log)
    assert res["status"] == "failed"
    assert "error" in res["reason"]
