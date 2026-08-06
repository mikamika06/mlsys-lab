import torch

def measure_peak_memory(tensor_shape, chunk_size):
    torch.manual_seed(42)
    x = torch.randn(tensor_shape, dtype=torch.float32)
    chunks = torch.split(x, chunk_size, dim=0)
    naive_intermediates = [c * 2.0 for c in chunks]
    naive_peak = sum(t.element_size() * t.nelement() for t in naive_intermediates) + x.element_size() * x.nelement()
    fused_accum = torch.zeros((tensor_shape[0], tensor_shape[1]), dtype=torch.float32)
    current_peak = 0
    for c in chunks:
        res = c * 2.0
        fused_accum[0:res.shape[0], :] += res
        curr = res.element_size() * res.nelement() + x.element_size() * x.nelement()
        if curr > current_peak:
            current_peak = curr
    return float(current_peak), float(naive_peak)
