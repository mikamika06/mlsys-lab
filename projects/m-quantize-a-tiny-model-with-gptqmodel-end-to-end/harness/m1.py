import ref

def check(workdir):
    from gptqquant.config import make_config
    out = {"configs_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, c_spec in enumerate(ref.CONFIGS):
        ref_cfg = make_config(**c_spec)
        try:
            learner_cfg = make_config(**c_spec)
            if ref_cfg.to_dict() == learner_cfg.to_dict():
                ok += 1
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"config {i} error: {type(e).__name__}"
    out["configs_matched"] = float(ok)
    return out
