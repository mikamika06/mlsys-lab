import sys
import numpy as np

sys.path.insert(0, ".")
from imatrix_analysis.threshold import find_imatrix_convergence_bpw


def test_convergence_threshold():
    sample_data = [
        {
            "role": "attn_q",
            "weight": 1.0,
            "bpws": [2.0, 3.0, 4.0, 6.0, 8.0],
            "unweighted_errors": [0.50, 0.20, 0.08, 0.02, 0.005],
            "imatrix_errors":    [0.30, 0.10, 0.079, 0.0195, 0.0049],
        },
        {
            "role": "ffn_down",
            "weight": 2.0,
            "bpws": [2.0, 3.0, 4.0, 6.0, 8.0],
            "unweighted_errors": [0.60, 0.25, 0.10, 0.03, 0.008],
            "imatrix_errors":    [0.35, 0.12, 0.0995, 0.0299, 0.0079],
        }
    ]

    val = find_imatrix_convergence_bpw(sample_data, tol=1e-2)
    assert isinstance(val, float)
    assert 3.5 <= val <= 4.5, f"Expected convergence around 4.0 bpw, got {val}"


def test_convergence_threshold_strict_tol():
    sample_data = [
        {
            "role": "attn_k",
            "weight": 1.0,
            "bpws": [2.0, 3.0, 4.0, 6.0, 8.0],
            "unweighted_errors": [0.50, 0.20, 0.08, 0.02, 0.005],
            "imatrix_errors":    [0.30, 0.10, 0.05, 0.02, 0.005],
        }
    ]
    val = find_imatrix_convergence_bpw(sample_data, tol=1e-4)
    assert val == 6.0, f"Expected convergence at 6.0 bpw, got {val}"
