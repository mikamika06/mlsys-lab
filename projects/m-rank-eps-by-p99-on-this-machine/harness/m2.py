import ref


def check(workdir):
    from ep_selector.fragmentation import compute_fragmentation_cost
    from ep_selector.policy import select_best_backend

    _, nodes, supported = ref.get_test_data()

    want_frag = ref.compute_fragmentation_cost(nodes, supported)
    want_policy1 = ref.select_best_backend(["tensorrt", "ort-cuda"], want_frag, 0.2)
    want_policy2 = ref.select_best_backend(["tensorrt", "ort-cuda"], want_frag, 0.9)

    try:
        got_frag = compute_fragmentation_cost(nodes, supported)
        got_policy1 = select_best_backend(["tensorrt", "ort-cuda"], got_frag, 0.2)
        got_policy2 = select_best_backend(["tensorrt", "ort-cuda"], got_frag, 0.9)
    except Exception as e:
        return {
            "fragmentation_matched": 0.0,
            "churn_policy_matched": 0.0,
            "_note": f"raised {type(e).__name__}: {e}"
        }

    frag_ok = 1.0 if abs(got_frag - want_frag) < 1e-5 else 0.0
    policy_ok = 1.0 if (got_policy1 == want_policy1 and got_policy2 == want_policy2) else 0.0

    out = {
        "fragmentation_matched": frag_ok,
        "churn_policy_matched": policy_ok
    }
    if frag_ok == 0.0:
        out["_note"] = f"fragmentation cost got {got_frag}, want {want_frag}"
    elif policy_ok == 0.0:
        out["_note"] = f"policy got ({got_policy1}, {got_policy2}), want ({want_policy1}, {want_policy2})"
    return out
