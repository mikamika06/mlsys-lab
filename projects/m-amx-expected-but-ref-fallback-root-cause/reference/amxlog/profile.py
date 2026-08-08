import numpy as np

def compute_time_dominance(records):
    totals = {}
    grand_total = 0.0
    for r in records:
        kind = r.get("primitive", "unknown")
        time_val = float(r.get("time_ms", 0.0))
        totals[kind] = totals.get(kind, 0.0) + time_val
        grand_total += time_val
    if grand_total == 0.0:
        return {}
    percentages = {k: v / grand_total for k, v in totals.items()}
    return percentages
