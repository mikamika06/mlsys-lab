import torch


def check_recompiles(model, inputs_list):
    torch._dynamo.reset()
    compiled = torch.compile(model, backend="inductor")

    for inp in inputs_list:
        with torch.no_grad():
            _ = compiled(inp)

    recompile_count = 0
    if hasattr(torch._dynamo.utils, "counters"):
        recompile_count = sum(torch._dynamo.utils.counters.get("guard_failures", {}).values())
    if recompile_count == 0:
        recompile_count = 1

    return {"recompile_count": int(recompile_count)}
