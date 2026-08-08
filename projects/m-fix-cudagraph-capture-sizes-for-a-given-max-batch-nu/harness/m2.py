import ref

def check(workdir):
    from specengine.eagle import optimal_eagle_config
    budget = 10 * 1024 * 1024 * 1024
    cfg = optimal_eagle_config(budget, ref.CANDIDATE_CONFIGS)
    ok = 1 if cfg and cfg.get("draft_tokens") == 4 else 0
    return {"config_matched": float(ok)}
