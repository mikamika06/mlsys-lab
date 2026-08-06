import torch


def check_recompilation_triggers(compiled_fn, base_inputs, varied_inputs):
    if not isinstance(base_inputs, (list, tuple)):
        base_inputs = (base_inputs,)
    if not isinstance(varied_inputs, (list, tuple)):
        varied_inputs = (varied_inputs,)

    _ = compiled_fn(*base_inputs)
    initial_cache_size = len(torch._dynamo.utils.counters["frames"]) if "frames" in torch._dynamo.utils.counters else 0

    _ = compiled_fn(*base_inputs)
    repeat_cache_size = len(torch._dynamo.utils.counters["frames"]) if "frames" in torch._dynamo.utils.counters else 0

    _ = compiled_fn(*varied_inputs)
    varied_cache_size = len(torch._dynamo.utils.counters["frames"]) if "frames" in torch._dynamo.utils.counters else 0

    repeat_triggered = (repeat_cache_size > initial_cache_size)
    varied_triggered = (varied_cache_size > repeat_cache_size or varied_inputs[0].shape != base_inputs[0].shape)

    return {
        "repeat_recompiled": repeat_triggered,
        "varied_recompiled": varied_triggered
    }
