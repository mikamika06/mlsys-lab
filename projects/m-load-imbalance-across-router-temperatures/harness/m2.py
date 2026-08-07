import ref
import numpy as np
import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from moe_routing.capacity import optimal_capacity_factor
    from moe_routing.comm import all_to_all_shapes

    out = {"capacity_match": 0.0, "shapes_match": 0.0}
    try:
        assigns = ref.assign_tokens(ref.LOGITS, 1.0, 0.1)
        num_tokens = ref.LOGITS.shape[0]
        num_experts = 8

        for max_drop in [0.05, 0.1, 0.2]:
            got_cap = optimal_capacity_factor(assigns, num_experts, num_tokens, max_drop)
            want_cap = ref.optimal_capacity_factor(assigns, num_experts, num_tokens, max_drop)
            if abs(got_cap - want_cap) > 1e-4:
                out["_note"] = f"capacity mismatch for max_drop {max_drop}"
                return out
        out["capacity_match"] = 1.0

        for ndevs in [2, 4, 8]:
            got_s, got_r = all_to_all_shapes(assigns, num_experts, ndevs)
            want_s, want_r = ref.all_to_all_shapes(assigns, num_experts, ndevs)
            if not (np.array_equal(got_s, want_s) and np.array_equal(got_r, want_r)):
                out["_note"] = f"shapes mismatch for {ndevs} devices"
                return out
        out["shapes_match"] = 1.0
    except Exception as e:
        out["_note"] = f"Error: {e}"
    return out
