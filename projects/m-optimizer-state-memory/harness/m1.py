import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    out = {"state_memory_matched": 0.0}

    try:
        from optmem.state import estimate_optimizer_state_bytes, calculate_model_optimizer_footprint
    except Exception as e:
        out["_note"] = f"Failed to import optmem.state: {e}"
        return out

    model = ref.get_test_model()
    params = list(model.parameters())
    opts = ["sgd", "momentum", "adam", "adamw"]

    ok = True
    for opt in opts:
        want_est = ref.ref_estimate_optimizer_state_bytes(params, opt, initialized=True)
        got_est = estimate_optimizer_state_bytes(params, opt, initialized=True)
        if want_est != got_est:
            ok = False
            out["_note"] = f"Mismatch for {opt}: want {want_est}, got {got_est}"
            break

        want_uninit = ref.ref_estimate_optimizer_state_bytes(params, opt, initialized=False)
        got_uninit = estimate_optimizer_state_bytes(params, opt, initialized=False)
        if want_uninit != got_uninit:
            ok = False
            out["_note"] = f"Uninit mismatch for {opt}: want {want_uninit}, got {got_uninit}"
            break

    if ok:
        want_fp = ref.ref_calculate_model_optimizer_footprint(params, opts)
        got_fp = calculate_model_optimizer_footprint(params, opts)
        if want_fp == got_fp:
            out["state_memory_matched"] = 1.0
        else:
            out["_note"] = f"Footprint mismatch: want {want_fp}, got {got_fp}"

    return out
