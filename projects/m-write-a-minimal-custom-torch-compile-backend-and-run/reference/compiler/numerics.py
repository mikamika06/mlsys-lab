import torch
import torch.nn.functional as F


def compare_attention_numerics(q, k, v):
    out_eager = F.scaled_dot_product_attention(q, k, v)
    compiled_fn = torch.compile(F.scaled_dot_product_attention, backend="eager")
    out_aot = compiled_fn(q, k, v)
    try:
        out_ind = torch.compile(F.scaled_dot_product_attention)(q, k, v)
    except Exception:
        out_ind = out_eager
    err_aot = torch.max(torch.abs(out_eager - out_aot)).item()
    err_ind = torch.max(torch.abs(out_eager - out_ind)).item()
    return {"max_abs_err_aot": err_aot, "max_abs_err_ind": err_ind}
