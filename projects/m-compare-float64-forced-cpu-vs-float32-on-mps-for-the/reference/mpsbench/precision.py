import torch


def compare_precision(a_cpu64, b_cpu64, fn):
    out_cpu64 = fn(a_cpu64, b_cpu64)
    dev = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    a_f32 = a_cpu64.to(dtype=torch.float32, device=dev)
    b_f32 = b_cpu64.to(dtype=torch.float32, device=dev)
    out_f32 = fn(a_f32, b_f32).to(dtype=torch.float64, device="cpu")
    diff = torch.norm(out_cpu64 - out_f32)
    ref_norm = torch.norm(out_cpu64)
    rel_err = float(diff / (ref_norm + 1e-12))
    return out_cpu64, out_f32, rel_err
