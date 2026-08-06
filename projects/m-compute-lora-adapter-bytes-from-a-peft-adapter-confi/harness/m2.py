import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from loraserve.budget import compute_preallocated_budget, can_fit_adapters
    from loraserve.scheduler import schedule_adapter_batch
    import ref as reference_impl

    from loraserve.budget import compute_preallocated_budget as learner_budget
    from loraserve.budget import can_fit_adapters as learner_can_fit
    from loraserve.scheduler import schedule_adapter_batch as learner_schedule

    out = {
        "budget_matched": 0.0,
        "fit_matched": 0.0,
        "schedule_matched": 0.0
    }

    b_got = learner_budget(
        num_layers=32,
        target_modules=["q_proj", "v_proj"],
        hidden_shapes={"q_proj": (4096, 4096), "v_proj": (4096, 4096)},
        max_loras=8,
        max_lora_rank=16,
        dtype_bytes=2
    )
    b_want = compute_preallocated_budget(
        num_layers=32,
        target_modules=["q_proj", "v_proj"],
        hidden_shapes={"q_proj": (4096, 4096), "v_proj": (4096, 4096)},
        max_loras=8,
        max_lora_rank=16,
        dtype_bytes=2
    )
    if b_got == b_want:
        out["budget_matched"] = 1.0
    else:
        out["_note"] = f"Budget mismatch: want {b_want}, got {b_got}"

    adapters = [t["peft"] for t in reference_impl.CONFIG_TEMPLATES]
    base_shapes = reference_impl.CONFIG_TEMPLATES[0]["shapes"]

    cap_tight = 10_000_000
    cap_generous = 500_000_000

    fit_tight_got = learner_can_fit(adapters, base_shapes, cap_tight)
    fit_tight_want = can_fit_adapters(adapters, base_shapes, cap_tight)
    fit_gen_got = learner_can_fit(adapters, base_shapes, cap_generous)
    fit_gen_want = can_fit_adapters(adapters, base_shapes, cap_generous)

    if fit_tight_got == fit_tight_want and fit_gen_got == fit_gen_want:
        out["fit_matched"] = 1.0
    elif "_note" not in out:
        out["_note"] = f"can_fit mismatch: tight ({fit_tight_got} vs {fit_tight_want}), generous ({fit_gen_got} vs {fit_gen_want})"

    requests = reference_impl.generate_workload(seed=123)
    sched_got = learner_schedule(requests, max_batch_size=4, max_active_adapters=2)
    sched_want = schedule_adapter_batch(requests, max_batch_size=4, max_active_adapters=2)

    if sched_got == sched_want:
        out["schedule_matched"] = 1.0
    elif "_note" not in out:
        out["_note"] = f"Schedule mismatch on workload. Batches got={len(sched_got)}, want={len(sched_want)}"

    return out
