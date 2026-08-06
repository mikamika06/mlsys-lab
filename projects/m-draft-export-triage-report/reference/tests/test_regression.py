"""Regression tests for draft export triage."""
from triage.report import generate_triage_report


def test_triage_unclassified_detection():
    sample_results = [
        {
            "model_name": "test_m",
            "success": False,
            "issues": [
                {"node": "node_0", "code": "RARE_INTERNAL_ERR", "message": "Unexpected failure"}
            ],
            "node_count": 5
        }
    ]
    report = generate_triage_report(sample_results)
    assert report["issue_counts"]["UNKNOWN"] == 1
    assert report["total_issues"] == 1
