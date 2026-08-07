def check(workdir):
    from moe.cache import ExpertCache
    from moe.prefetch import prefetch_plan

    m = {"prefetch_coverage": 0.0}
    cache = ExpertCache(2000)
    decisions = [{"next_experts": [1, 2]}]
    plan = prefetch_plan(decisions, cache, 100)
    if set(plan) == {1, 2}:
        m["prefetch_coverage"] = 1.0
    return m
