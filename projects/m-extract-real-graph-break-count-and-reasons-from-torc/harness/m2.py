import ref
import torch


def check(workdir):
    from peftcomp.warmup import measure_warmup_cost
    model, x_base, _ = ref.get_oracle_model_and_inputs()
    compiled = torch.compile(model)
    res = measure_warmup_cost(compiled, x_base)
    measured = 1.0 if isinstance(res, dict) and "warmup_gap_ns" in res and res["warmup_gap_ns"] >= 0 else 0.0
    out = {"warmup_measured": measured}
    if measured == 0.0:
        out["_note"] = f"Invalid warmup measurement structure or negative gap: {res}"
    return out
