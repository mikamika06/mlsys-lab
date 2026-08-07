import sys
sys.path.insert(0, ".")
from tritonval.validator import validate_config
from tritonval.layout import classify_layout
from tritonval.versioning import compute_resident_versions

def test_validate_config_missing_name():
    cfg = 'platform: "tensorrt_plan"\nmax_batch_size: 4'
    res = validate_config(cfg)
    assert not res["valid"]
    assert any("name" in e for e in res["errors"])

def test_validate_config_valid():
    cfg = 'name: "resnet"\nplatform: "tensorrt_plan"\nmax_batch_size: 8'
    res = validate_config(cfg)
    assert res["valid"]
    assert len(res["errors"]) == 0

def test_versioning_latest_policy():
    available = [1, 2, 3, 10]
    policy = {"latest": {"count": 2}}
    resident = compute_resident_versions(available, policy)
    assert resident == [3, 10]
