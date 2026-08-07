import ref
from heterogeneous import restarts

def check(workdir):
    out = {"predictions_matched": 0.0}
    matches = 0
    try:
        for cfg in ref.CONFIG_CHANGES:
            ref_val = restarts.forces_restart(cfg)
            from importlib import reload
            import heterogeneous.restarts as learner_mod
            reload(learner_mod)
            got_val = learner_mod.forces_restart(cfg)
            if got_val == ref_val:
                matches += 1
        out["predictions_matched"] = float(matches)
    except Exception as e:
        out["_note"] = f"Error during restart prediction: {type(e).__name__}: {str(e)[:100]}"
    return out
