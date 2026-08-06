"""Report generator module."""


def generate_triage_report(triage_results):
    total_models = len(triage_results)
    successful = sum(1 for r in triage_results if r.get("success", False))
    failed = total_models - successful

    code_counts = {
        "GUARD_VIOLATION": 0,
        "SIDE_EFFECT": 0,
        "UNSUPPORTED_OP": 0,
        "UNKNOWN": 0
    }

    for r in triage_results:
        for issue in r.get("issues", []):
            code = issue.get("code", "UNKNOWN")
            if code in code_counts:
                code_counts[code] += 1
            else:
                code_counts["UNKNOWN"] += 1

    total_issues = sum(code_counts.values())

    weights = {
        "GUARD_VIOLATION": 2,
        "SIDE_EFFECT": 5,
        "UNSUPPORTED_OP": 10,
        "UNKNOWN": 8
    }

    priority_score = sum(code_counts[k] * weights[k] for k in weights)

    if priority_score == 0:
        status = "READY"
    elif priority_score < 15:
        status = "NEEDS_ANNOTATION"
    else:
        status = "REQUIRES_REWRITE"

    return {
        "total_models": total_models,
        "successful_exports": successful,
        "failed_exports": failed,
        "total_issues": total_issues,
        "issue_counts": code_counts,
        "priority_score": priority_score,
        "status": status
    }
