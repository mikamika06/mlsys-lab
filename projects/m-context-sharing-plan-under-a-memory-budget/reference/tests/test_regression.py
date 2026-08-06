import sys

sys.path.insert(0, ".")
from ctxplan.plan import build_sharing_plan
from ctxplan.classifier import classify_oom
from ctxplan.evidence import extract_exclusion_evidence


def test_sharing_plan_respects_budget():
    tensors = [{"id": 1, "size": 1000}, {"id": 2, "size": 2000}]
    res = build_sharing_plan(tensors, 1500)
    assert sum(t["size"] for t in tensors if t["id"] in res) <= 1500


def test_classifier_detects_build_oom():
    log = {"phase": "build", "peak_memory": 500}
    assert classify_oom(log, 1000) == "build"


def test_evidence_extraction():
    logs = ["Tactic 42 rejected: insufficient workspace"]
    res = extract_exclusion_evidence(logs)
    assert len(res) > 0
