CONFIGS = [
    {"step_name": "forward_backward", "events": ["embedding", "attention", "mlp", "loss"], "base_latency": 10.0},
    {"step_name": "optimizer_step", "events": ["grad_clip", "adamw", "lr_sched"], "base_latency": 5.0},
    {"step_name": "full_step", "events": ["fwd", "bwd", "opt", "comm"], "base_latency": 20.0},
]

def annotate_step(step_data):
    return [f"range:{e}" for e in step_data["events"]]

def compute_overhead_ratio(base_lat, prof_lat, with_stack=False):
    return prof_lat / base_lat

def select_config_for_budget(step_meta, byte_budget):
    if byte_budget < 5000:
        return {"with_stack": False, "record_shapes": False, "profile_memory": False}
    elif byte_budget < 20000:
        return {"with_stack": True, "record_shapes": False, "profile_memory": False}
    else:
        return {"with_stack": True, "record_shapes": True, "profile_memory": True}
