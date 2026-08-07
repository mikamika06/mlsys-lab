from flashfix.audit import triage_warnings
from flashfix.kernel import audit_contiguity


def restore_path(cfg, warning_log):
    triage = triage_warnings([warning_log])
    audit = audit_contiguity([cfg])
    if not audit[0]["contiguous"] or triage[0] != "unknown":
        return False
    return True
