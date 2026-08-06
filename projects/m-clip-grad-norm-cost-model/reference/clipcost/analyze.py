from clipcost.model import ClipCostModel


def estimate_clip_cost(parameters, bandwidth_gbps: float) -> dict:
    model = ClipCostModel(parameters)
    tot = model.total_bytes()
    t_us = model.estimated_time_us(bandwidth_gbps)
    return {
        "total_bytes": tot,
        "estimated_time_us": t_us,
        "group_count": len(model.groups)
    }
