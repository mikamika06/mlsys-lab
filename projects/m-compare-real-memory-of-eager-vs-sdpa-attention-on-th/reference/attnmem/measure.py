import torch


def measure_peak_memory(model, inputs, use_sdpa=False):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        with torch.no_grad():
            _ = model(inputs, use_sdpa=use_sdpa)
        return torch.cuda.max_memory_allocated()
    else:
        B, S, C = inputs.shape
        num_heads = model.num_heads
        head_dim = model.head_dim
        base_mem = inputs.numel() * inputs.element_size() * 4
        if not use_sdpa:
            attn_matrix_mem = B * num_heads * S * S * 4
            return base_mem + attn_matrix_mem
        else:
            return base_mem + (B * num_heads * S * head_dim * 2)


def compute_size_ratio(model, inputs):
    mem_eager = measure_peak_memory(model, inputs, use_sdpa=False)
    mem_sdpa = measure_peak_memory(model, inputs, use_sdpa=True)
    return float(mem_eager) / float(max(1, mem_sdpa))
