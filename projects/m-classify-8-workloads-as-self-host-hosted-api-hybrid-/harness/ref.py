WORKLOADS = [
    {
        "id": "w1",
        "name": "Global Support Chatbot",
        "avg_qps": 45.0,
        "peak_qps": 200.0,
        "data_privacy": "standard",
        "requires_finetuning": False,
        "monthly_tokens": 5000000000
    },
    {
        "id": "w2",
        "name": "Medical Records Summarizer",
        "avg_qps": 2.0,
        "peak_qps": 12.0,
        "data_privacy": "hipaa_strict",
        "requires_finetuning": False,
        "monthly_tokens": 150000000
    },
    {
        "id": "w3",
        "name": "Internal Code Assistant",
        "avg_qps": 15.0,
        "peak_qps": 80.0,
        "data_privacy": "proprietary",
        "requires_finetuning": True,
        "monthly_tokens": 1200000000
    },
    {
        "id": "w4",
        "name": "E-Commerce Search Re-ranker",
        "avg_qps": 120.0,
        "peak_qps": 600.0,
        "data_privacy": "standard",
        "requires_finetuning": False,
        "monthly_tokens": 15000000000
    },
    {
        "id": "w5",
        "name": "Spiky Marketing Copy Generator",
        "avg_qps": 0.5,
        "peak_qps": 90.0,
        "data_privacy": "standard",
        "requires_finetuning": False,
        "monthly_tokens": 80000000
    },
    {
        "id": "w6",
        "name": "Legal Contract Analyzer",
        "avg_qps": 1.0,
        "peak_qps": 5.0,
        "data_privacy": "strict",
        "requires_finetuning": False,
        "monthly_tokens": 50000000
    },
    {
        "id": "w7",
        "name": "Burst-Heavy Document Extractor",
        "avg_qps": 8.0,
        "peak_qps": 250.0,
        "data_privacy": "standard",
        "requires_finetuning": False,
        "monthly_tokens": 900000000
    },
    {
        "id": "w8",
        "name": "General Analytics Copilot",
        "avg_qps": 25.0,
        "peak_qps": 90.0,
        "data_privacy": "standard",
        "requires_finetuning": False,
        "monthly_tokens": 2500000000
    }
]

def classify_workload(w):
    if w["data_privacy"] in ("hipaa_strict", "strict", "proprietary") and w["requires_finetuning"]:
        return {"deployment": "self-host", "deciding_factor": "strict_privacy_and_finetuning"}
    if w["data_privacy"] in ("hipaa_strict", "strict", "proprietary"):
        return {"deployment": "self-host", "deciding_factor": "strict_data_governance"}
    if w["requires_finetuning"]:
        return {"deployment": "self-host", "deciding_factor": "custom_weights_finetuning"}
    
    ratio = w["peak_qps"] / max(0.1, w["avg_qps"])
    if ratio > 15.0 and w["monthly_tokens"] > 500000000:
        return {"deployment": "hybrid", "deciding_factor": "bursty_traffic_with_high_baseline"}
    
    if w["monthly_tokens"] > 2000000000 or w["avg_qps"] > 20.0:
        return {"deployment": "self-host", "deciding_factor": "high_volume_tco_advantage"}
    
    if w["monthly_tokens"] < 200000000 and w["avg_qps"] < 5.0:
        return {"deployment": "hosted-api", "deciding_factor": "low_volume_api_cost_efficiency"}
    
    return {"deployment": "hosted-api", "deciding_factor": "moderate_volume_api_convenience"}

def get_reference_classifications():
    res = {}
    for w in WORKLOADS:
        res[w["id"]] = classify_workload(w)
    return res

def compute_metrics(workloads):
    res = {}
    for w in workloads:
        c = classify_workload(w)
        dep = c["deployment"]
        tokens = w["monthly_tokens"]
        if dep == "self-host":
            cost = 1500.0 + tokens * 0.0000001
        elif dep == "hosted-api":
            cost = tokens * 0.0000015
        else:
            cost = 800.0 + tokens * 0.0000008
        res[w["id"]] = {"deployment": dep, "estimated_monthly_cost": round(cost, 2)}
    return res
