import ref
import torch

def check(workdir):
    from gradckpt.memory import measure_peak_memory
    torch.manual_seed(42)
    model = ref.ToyModel(hidden_dim=32, num_layers=4)
    x = torch.randn(2, 32)
    got = measure_peak_memory(model, x)
    out = {"memory_measured": 0.0}
    if isinstance(got, dict) and "mem_with" in got and "mem_without" in got:
        if got["mem_with"] < got["mem_without"]:
            out["memory_measured"] = 1.0
        else:
            out["_note"] = f"mem_with ({got['mem_with']}) not less than mem_without ({got['mem_without']})"
    else:
        out["_note"] = "measure_peak_memory must return a dict with mem_with and mem_without"
    return out
