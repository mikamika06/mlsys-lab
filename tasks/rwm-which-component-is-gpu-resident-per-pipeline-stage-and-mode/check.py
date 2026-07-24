def _reference(mode):
    stages = ["encode", "denoise", "decode"]
    if mode == "model":
        comp_set = {"text_encoder", "denoiser", "vae"}
        return {s: set(comp_set) for s in stages}
    elif mode == "sequential":
        mapping = {
            "encode": {"text_encoder"},
            "denoise": {"denoiser"},
            "decode": {"vae"},
        }
        return mapping
    elif mode == "group":
        mapping = {
            "encode": {"text_encoder", "denoiser"},
            "denoise": {"denoiser", "vae"},
            "decode": {"vae"},
        }
        return mapping
    else:
        raise ValueError(f"Unknown mode: {mode}")

def grade(sol, fx) -> dict:
    modes = ["model", "sequential", "group"]
    ok = 1.0
    for m in modes:
        try:
            got = sol.residency(m)
            ref = _reference(m)
        except Exception:
            return {"exact_match": 0.0}
        # Normalize: convert values to sets if they aren't already
        try:
            got_norm = {k: set(v) for k, v in got.items()}
        except Exception:
            return {"exact_match": 0.0}
        if got_norm != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
