def estimate_bandwidth_waste(costs):
    materialized = costs.get("materialized_bytes", 0)
    score = costs.get("lost_fusion_score", 0)
    penalty_factor = 2.5
    estimated_waste_gbps = float(materialized * score * penalty_factor) / 1e9
    return {"bandwidth_waste_score": estimated_waste_gbps}
