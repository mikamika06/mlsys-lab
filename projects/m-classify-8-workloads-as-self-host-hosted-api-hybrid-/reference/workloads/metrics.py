"""TCO and cost metrics module."""
from workloads.classifier import classify_workload

def compute_tco_metrics(workloads):
    res = {}
    for w in workloads:
        c = classify_workload(w)
        dep = c["deployment"]
        tokens = w.get("monthly_tokens", 0)
        if dep == "self-host":
            cost = 1500.0 + tokens * 0.0000001
        elif dep == "hosted-api":
            cost = tokens * 0.0000015
        else:
            cost = 800.0 + tokens * 0.0000008
        res[w["id"]] = {"deployment": dep, "estimated_monthly_cost": round(cost, 2)}
    return res
