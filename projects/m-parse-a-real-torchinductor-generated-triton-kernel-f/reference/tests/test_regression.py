import sys
sys.path.insert(0, ".")
from inductor_parse.parser import parse_kernel_config
from inductor_parse.diff import diff_configs
from inductor_parse.autotune import find_argmin_config

SAMPLE_KERNEL = """
@triton_heuristics.template(
    configs=[
        transform_config({'XBLOCK': 64, 'RBLOCK': 128}, num_warps=8, num_stages=4),
    ],
)
@triton.jit
def triton_k(in_ptr, out_ptr, XBLOCK : tl.constexpr):
    pass
"""


def test_parse_kernel_config_extracts_all_parameters():
    cfg = parse_kernel_config(SAMPLE_KERNEL)
    assert cfg["XBLOCK"] == 64
    assert cfg["RBLOCK"] == 128
    assert cfg["num_warps"] == 8
    assert cfg["num_stages"] == 4


def test_diff_configs_identifies_changes():
    d1 = {"XBLOCK": 32, "num_warps": 4}
    d2 = {"XBLOCK": 64, "num_warps": 4}
    res = diff_configs(d1, d2)
    assert "XBLOCK" in res["changed"]
    assert res["changed"]["XBLOCK"] == {"default": 32, "autotune": 64}
    assert res["same"] == {"num_warps": 4}


def test_find_argmin_config_selects_fastest_ok_candidate():
    logs = [
        {"config": {"XBLOCK": 32}, "time_ms": 1.5, "status": "OK"},
        {"config": {"XBLOCK": 64}, "time_ms": 0.8, "status": "OK"},
        {"config": {"XBLOCK": 128}, "time_ms": 0.2, "status": "FAILED"},
    ]
    res = find_argmin_config(logs)
    assert res["config"] == {"XBLOCK": 64}
    assert res["time_ms"] == 0.8
    assert res["num_candidates_evaluated"] == 3
