"""Workload classification module."""

def classify_workload(w):
    if w.get("data_privacy") in ("hipaa_strict", "strict", "proprietary") and w.get("requires_finetuning"):
        return {"deployment": "self-host", "deciding_factor": "strict_privacy_and_finetuning"}
    if w.get("data_privacy") in ("hipaa_strict", "strict", "proprietary"):
        return {"deployment": "self-host", "deciding_factor": "strict_data_governance"}
    if w.get("requires_finetuning"):
        return {"deployment": "self-host", "deciding_factor": "custom_weights_finetuning"}
    
    peak = w.get("peak_qps", 0.0)
    avg = w.get("avg_qps", 0.1)
    ratio = peak / max(0.1, avg)
    tokens = w.get("monthly_tokens", 0)

    if ratio > 15.0 and tokens > 500000000:
        return {"deployment": "hybrid", "deciding_factor": "bursty_traffic_with_high_baseline"}
    if tokens > 2000000000 or avg > 20.0:
        return {"deployment": "self-host", "deciding_factor": "high_volume_tco_advantage"}
    if tokens < 200000000 and avg < 5.0:
        return {"deployment": "hosted-api", "deciding_factor": "low_volume_api_cost_efficiency"}
    return {"deployment": "hosted-api", "deciding_factor": "moderate_volume_api_convenience"}

def classify_all(workloads):
    return {w["id"]: classify_workload(w) for w in workloads}
