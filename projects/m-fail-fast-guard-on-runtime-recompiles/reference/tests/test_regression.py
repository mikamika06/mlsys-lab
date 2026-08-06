import sys
sys.path.insert(0, ".")
from recompile.policy import lookup_policy


def test_policy_lookup_exact():
    tbl = {("strict", False): "fail_fast"}
    assert lookup_policy(tbl, ("strict", False)) == "fail_fast"
