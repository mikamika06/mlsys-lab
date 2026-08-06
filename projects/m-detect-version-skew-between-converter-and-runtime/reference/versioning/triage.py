def triage_skew_report(report):
    if not report.get("has_skew", False):
        return {
            "severity": "COMPATIBLE",
            "critical_count": 0,
            "warning_count": 0
        }

    criticals = 0
    warnings = 0

    if report.get("unsupported_quant_types"):
        criticals += len(report["unsupported_quant_types"])

    if report.get("missing_required_keys"):
        criticals += len(report["missing_required_keys"])

    mismatches = report.get("version_mismatches", [])
    for m in mismatches:
        if "Version mismatch" in m:
            criticals += 1
        else:
            warnings += 1

    severity = "FATAL" if criticals > 0 else "WARNING"

    return {
        "severity": severity,
        "critical_count": criticals,
        "warning_count": warnings
    }


def suggest_remediation(triage_result):
    sev = triage_result.get("severity")
    if sev == "COMPATIBLE":
        return "No action required. Converter and runtime are compatible."
    elif sev == "WARNING":
        return "Proceed with caution. Non-critical version metadata mismatches detected."
    else:
        return "Upgrade runtime or re-convert model using matching toolchain version."
