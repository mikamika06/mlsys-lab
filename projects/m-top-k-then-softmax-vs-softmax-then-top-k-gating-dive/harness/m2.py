import numpy as np
import ref


def check(workdir):
    from moegating.dispatch import build_mixtral_dispatch_tensor
    from moegating.gating import top_k_then_softmax
    from moegating.metrics import compute_router_entropy

    out = {
        "max_abs_err": 0.0,
        "dispatch_matches": 0.0,
        "entropy_matches": 0.0,
    }

    logits = ref.LOGITS_DATA[0]
    _, selected_experts = top_k_then_softmax(logits, ref.TOP_K)

    want_dispatch = ref.build_mixtral_dispatch_tensor(
        selected_experts, ref.NUM_EXPERTS
    )
    got_dispatch = build_mixtral_dispatch_tensor(selected_experts, ref.NUM_EXPERTS)

    if not np.array_equal(want_dispatch, got_dispatch):
        out["_note"] = "dispatch tensor layout or values mismatch"
        out["max_abs_err"] = 1.0
        return out
    out["dispatch_matches"] = 1.0

    want_entropy = ref.compute_router_entropy(ref.LOGITS_DATA)
    got_entropy = compute_router_entropy(ref.LOGITS_DATA)

    err1 = np.max(
        np.abs(
            want_entropy["mean_entropy_per_layer"]
            - got_entropy["mean_entropy_per_layer"]
        )
    )
    err2 = np.max(
        np.abs(
            want_entropy["entropy_per_token"] - got_entropy["entropy_per_token"]
        )
    )

    max_err = float(max(err1, err2))
    out["max_abs_err"] = max_err
    if max_err <= 1e-5:
        out["entropy_matches"] = 1.0

    return out
