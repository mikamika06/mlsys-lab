import sys
sys.path.insert(0, ".")
from auditor.packaging import configure_offline_env
from auditor.budget import predict_ready_time
from auditor.layout import audit_reused_blobs

def test_offline_env_variable_is_set():
    env = configure_offline_env()
    assert env.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE must be set to 1"

def test_budget_non_negative():
    val = predict_ready_time(100, 500, 1.0, 10, 50)
    assert val >= 0.0, "ready time cannot be negative"

def test_layout_auditor_exact():
    r = audit_reused_blobs({"a": 1}, {"a": 1, "b": 2})
    assert "a" in r["reused"]
    assert "b" in r["only_rev2"]
