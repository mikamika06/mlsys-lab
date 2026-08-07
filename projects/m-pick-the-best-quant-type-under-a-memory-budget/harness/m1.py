import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from quantplan.picker import find_best_quant_index

    out = {"argmin_index": 0.0}
    total = len(ref.SELECTION_SCENARIOS)
    matched = 0
    for i, sc in enumerate(ref.SELECTION_SCENARIOS):
        want_idx = ref.find_best_quant_index(
            sc["candidates"],
            sc["num_params"],
            sc["overhead_bytes"],
            sc["vram_budget_bytes"],
            sc["backend_config"],
            sc.get("allow_cpu_fallback", False),
        )
        got_idx = find_best_quant_index(
            sc["candidates"],
            sc["num_params"],
            sc["overhead_bytes"],
            sc["vram_budget_bytes"],
            sc["backend_config"],
            sc.get("allow_cpu_fallback", False),
        )
        if got_idx == want_idx:
            matched += 1
        elif "_note" not in out:
            out["_note"] = f"scenario {i}: got index {got_idx}, expected {want_idx}"

    if matched == total:
        out["argmin_index"] = 1.0
    return out
