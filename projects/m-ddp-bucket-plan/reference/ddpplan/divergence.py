def check_rank_plan_consistency(rank_plans):
    if not rank_plans:
        return {"consistent": True, "mismatched_ranks": []}
    ranks = sorted(rank_plans.keys())
    ref_rank = ranks[0]
    ref_plan = rank_plans[ref_rank]
    mismatched = []
    for r in ranks[1:]:
        if rank_plans[r] != ref_plan:
            mismatched.append(r)
    return {
        "consistent": len(mismatched) == 0,
        "mismatched_ranks": mismatched
    }
