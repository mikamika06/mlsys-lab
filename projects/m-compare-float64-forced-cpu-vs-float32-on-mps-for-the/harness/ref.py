import torch


def make_test_tensors(seed=42, size=(100, 100)):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(size, generator=g, dtype=torch.float64)
    b = torch.randn(size, generator=g, dtype=torch.float64)
    return a, b


def compute_ref_rel_err(a_cpu64, b_cpu64, fn):
    out_cpu64 = fn(a_cpu64, b_cpu64)
    a_f32 = a_cpu64.to(dtype=torch.float32)
    b_f32 = b_cpu64.to(dtype=torch.float32)
    out_f32 = fn(a_f32, b_f32).to(dtype=torch.float64)
    diff = torch.norm(out_cpu64 - out_f32)
    ref_norm = torch.norm(out_cpu64)
    return float(diff / (ref_norm + 1e-12))
