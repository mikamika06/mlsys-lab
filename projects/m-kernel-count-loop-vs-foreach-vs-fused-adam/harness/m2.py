import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)

    from opt.adam import step_loop, step_foreach, step_fused

    out = {
        "loop_matches_ref": 0.0,
        "foreach_matches_ref": 0.0,
        "fused_matches_ref": 0.0,
        "numerics_aligned": 0.0,
    }

    def copy_params_states(params, states):
        p_c = [{"id": p["id"], "device": p["device"], "dtype": p["dtype"],
                "param": p["param"].copy(), "grad": p["grad"].copy()} for p in params]
        s_c = [{"exp_avg": s["exp_avg"].copy(), "exp_avg_sq": s["exp_avg_sq"].copy(), "step": s["step"]} for s in states]
        return p_c, s_c

    params_base = ref.generate_synthetic_params()
    states_base = ref.generate_states(params_base)

    p_loop, s_loop = copy_params_states(params_base, states_base)
    p_foreach, s_foreach = copy_params_states(params_base, states_base)
    p_fused, s_fused = copy_params_states(params_base, states_base)

    try:
        for _ in range(3):
            step_loop(p_loop, s_loop, lr=1e-3, weight_decay=0.01)
        out["loop_matches_ref"] = 1.0
    except Exception as e:
        out["_note"] = f"step_loop failed: {e}"
        return out

    try:
        for _ in range(3):
            step_foreach(p_foreach, s_foreach, lr=1e-3, weight_decay=0.01)
        out["foreach_matches_ref"] = 1.0
    except Exception as e:
        out["_note"] = f"step_foreach failed: {e}"
        return out

    try:
        for _ in range(3):
            step_fused(p_fused, s_fused, lr=1e-3, weight_decay=0.01)
        out["fused_matches_ref"] = 1.0
    except Exception as e:
        out["_note"] = f"step_fused failed: {e}"
        return out

    loop_res = np.concatenate([p["param"].ravel() for p in p_loop])
    foreach_res = np.concatenate([p["param"].ravel() for p in p_foreach])
    fused_res = np.concatenate([p["param"].ravel() for p in p_fused])

    if np.allclose(loop_res, foreach_res, atol=1e-10) and np.allclose(foreach_res, fused_res, atol=1e-10):
        out["numerics_aligned"] = 1.0
    else:
        out["_note"] = "Numerical divergence detected across optimizer execution modes"

    return out
