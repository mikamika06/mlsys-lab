import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from qlora_mem.adamw import compute_adamw_state_bytes
        from qlora_mem.planner import qlora_peak_memory_plan
    except Exception as e:
        return {"adamw_paged_matched": 0.0, "planner_bytes_matched": 0.0, "fits_vram_matched": 0.0, "_note": f"Import error: {e}"}

    adamw_ok = True
    test_cases = [
        (1000, 256, False, 0),
        (1000, 256, True, 200),
        (20000000, 256, False, 650000),
        (20000000, 256, True, 650000),
    ]
    for n, b, p, m in test_cases:
        want = ref.compute_adamw_state_bytes(n, block_size=b, paged=p, max_layer_params=m)
        got = compute_adamw_state_bytes(n, block_size=b, paged=p, max_layer_params=m)
        if got != want:
            adamw_ok = False
            break

    configs = ref.generate_configs()
    plan_ok = True
    fits_ok = True
    for cfg in configs:
        want_plan = ref.qlora_peak_memory_plan(cfg)
        got_plan = qlora_peak_memory_plan(cfg)
        for k in ["base_weight_bytes", "lora_weight_bytes", "gradient_bytes", "optimizer_bytes", "activation_bytes", "workspace_bytes", "peak_vram_bytes"]:
            if got_plan.get(k) != want_plan[k]:
                plan_ok = False
                break
        if got_plan.get("fits_in_vram") != want_plan["fits_in_vram"]:
            fits_ok = False

    return {
        "adamw_paged_matched": 1.0 if adamw_ok else 0.0,
        "planner_bytes_matched": 1.0 if plan_ok else 0.0,
        "fits_vram_matched": 1.0 if fits_ok else 0.0,
    }
