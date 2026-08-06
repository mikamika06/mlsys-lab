import sys
sys.path.insert(0, ".")
from stallprof.analyzer import identify_imbalanced_branch
from stallprof.verifier import verify_sync_removal

def test_identify_imbalanced_branch_selects_highest_divergence():
    sample = {
        "kernel_id": "test_k",
        "warp_stats": {"stall_barrier": 1000},
        "branches": [
            {"branch_id": "low", "divergence_score": 10},
            {"branch_id": "high", "divergence_score": 90}
        ]
    }
    assert identify_imbalanced_branch(sample) == "high"

def test_verify_sync_removal_passes_valid_drop():
    base = {"warp_stats": {"stall_barrier": 5000}}
    mod = {"warp_stats": {"stall_barrier": 3000}}
    assert verify_sync_removal(base, mod, 1.20) is True
