import sys
sys.path.insert(0, ".")

from upgradeprep.diff import compute_upgrade_diff
from upgradeprep.checklist import generate_checklist

def test_breaking_changes_never_omitted():
    old_snap = {
        "flags": {"gpu_memory_utilization": "0.90"},
        "configs": {"max_model_len": "4096"},
        "deprecations": set(),
        "breaking": set()
    }
    new_snap = {
        "flags": {"gpu_memory_utilization": "0.90"},
        "configs": {"max_model_len": "8192"},
        "deprecations": set(),
        "breaking": {"vLLM Engine V1 API refactor"}
    }
    diff = compute_upgrade_diff(old_snap, new_snap)
    checklist = generate_checklist(diff)

    breaking_actions = [item["action"] for item in checklist if item["category"] == "BREAKING"]
    assert len(breaking_actions) == 1
    assert "vLLM Engine V1 API refactor" in breaking_actions[0]


def test_priority_ordering():
    diff = {
        "removed_flags": ["old_flag"],
        "added_flags": [],
        "changed_flags": {},
        "changed_configs": {},
        "new_deprecations": ["old_api"],
        "new_breaking": ["major_change"]
    }
    checklist = generate_checklist(diff)
    priorities = [item["priority"] for item in checklist]
    assert priorities == ["CRITICAL", "HIGH", "LOW"]
