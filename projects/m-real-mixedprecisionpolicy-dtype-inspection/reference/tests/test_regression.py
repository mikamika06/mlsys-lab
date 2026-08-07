import sys
sys.path.insert(0, ".")
from mpinspect.policy import inspect_policy
from mpinspect.sync import count_reduce_scatters
from mpinspect.accum import simulate_accumulation

def test_policy_inspection():
    p = {"param_dtype": "bf16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"}
    res = inspect_policy(p)
    assert res["param_dtype"] == "bf16"

def test_sync_counting():
    assert count_reduce_scatters(10, 4, True) == 3
    assert count_reduce_scatters(10, 4, False) == 10

def test_accumulation_behavior():
    val = simulate_accumulation(10, "fp32")
    assert val > 0.0
