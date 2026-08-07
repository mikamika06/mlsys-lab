def compute_residency_score(plan):
    total = sum(op["cost"] for op in plan["ops"])
    if total == 0:
        return 0.0
    ane = sum(op["cost"] for op in plan["ops"] if op["target"] == "ANE")
    return float(ane) / float(total)
