import ref
import numpy as np


def check(workdir):
    from ropescaling.scaling import compute_yarn_parameters, compute_llama3_scaling
    out = {"yarn_matched": 0.0, "llama3_matched": 0.0}
    yarn_ok = 0
    llama_ok = 0
    for cfg in ref.CONFIGS:
        w_freq, w_mscale = ref.compute_yarn_parameters(
            cfg["base"], cfg["seq_len"], cfg["max_pos"], cfg["orig_max"],
            cfg["beta_fast"], cfg["beta_slow"], cfg["mscale"]
        )
        g_freq, g_mscale = compute_yarn_parameters(
            cfg["base"], cfg["seq_len"], cfg["max_pos"], cfg["orig_max"],
            cfg["beta_fast"], cfg["beta_slow"], cfg["mscale"]
        )
        if np.allclose(g_freq, w_freq, atol=1e-4) and abs(g_mscale - w_mscale) < 1e-4:
            yarn_ok += 1

        w_llama = ref.compute_llama3_scaling(
            cfg["base"], cfg["seq_len"], cfg["max_pos"], cfg["orig_max"],
            cfg["factor"], cfg["low_freq"], cfg["high_freq"]
        )
        g_llama = compute_llama3_scaling(
            cfg["base"], cfg["seq_len"], cfg["max_pos"], cfg["orig_max"],
            cfg["factor"], cfg["low_freq"], cfg["high_freq"]
        )
        if np.allclose(g_llama, w_llama, atol=1e-4):
            llama_ok += 1

    out["yarn_matched"] = float(yarn_ok)
    out["llama3_matched"] = float(llama_ok)
    return out
