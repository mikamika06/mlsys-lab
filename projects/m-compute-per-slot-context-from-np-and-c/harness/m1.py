import sys

import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from llamaslot.context import compute_slot_context, plan_slot_allocation

    out = {
        "configs_matched": 0.0,
        "total_configs": float(len(ref.CONTEXT_CONFIGS)),
    }
    ok = 0
    for cfg in ref.CONTEXT_CONFIGS:
        want_slot = ref.compute_slot_context(
            cfg["requested_c"], cfg["requested_np"], cfg["model_max_ctx"]
        )
        got_slot = compute_slot_context(
            cfg["requested_c"], cfg["requested_np"], cfg["model_max_ctx"]
        )
        want_plan = ref.plan_slot_allocation(
            cfg["model_max_ctx"],
            cfg["requested_c"],
            cfg["requested_np"],
            cfg["min_tokens"],
        )
        got_plan = plan_slot_allocation(
            cfg["model_max_ctx"],
            cfg["requested_c"],
            cfg["requested_np"],
            cfg["min_tokens"],
        )
        if got_slot == want_slot and got_plan == want_plan:
            ok += 1
        elif "_note" not in out:
            out["_note"] = (
                f"cfg {cfg}: got slot {got_slot} (want {want_slot}), "
                f"got plan {got_plan} (want {want_plan})"
            )
    out["configs_matched"] = float(ok)
    return out
