import torch
from qlora.compare import compare_nf4_fp4


def test_quant_comparison_validity():
    torch.manual_seed(42)
    t = torch.randn(64, 64)
    res = compare_nf4_fp4(t)
    assert "nf4_mse" in res
    assert "fp4_mse" in res
    assert res["nf4_mse"] >= 0.0
    assert res["fp4_mse"] >= 0.0
    assert res["better"] in ("nf4", "fp4")
