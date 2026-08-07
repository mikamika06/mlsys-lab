import sys
sys.path.insert(0, ".")
from ampcheck.ops import classify_ops

def test_mm_is_fp16():
    policies = classify_ops()
    assert policies.get("aten.mm") == "fp16", "aten.mm should be in fp16 policy list"

def test_sum_is_promote():
    policies = classify_ops()
    assert policies.get("aten.sum") == "promote", "aten.sum should promote to fp32"

def test_all_ops_classified():
    policies = classify_ops()
    assert len(policies) == 15, "must classify exactly 15 operations"
