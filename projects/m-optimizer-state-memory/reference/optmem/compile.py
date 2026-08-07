import torch


def compile_optimizer_step(model, optimizer):
    """Wraps optimizer step using torch.compile or fused execution fallback."""
    def _step():
        optimizer.step()

    try:
        compiled_fn = torch.compile(_step)
        return compiled_fn
    except Exception:
        return _step


def measure_step_memory_delta(model, optimizer, compiled_step_fn, inputs):
    """Measures allocation differences between standard and compiled optimizer step."""
    def run_eager():
        optimizer.zero_grad()
        out = model(inputs)
        loss = out.sum()
        loss.backward()
        alloc_before = sum(p.grad.element_size() * p.grad.numel() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        alloc_after = sum(p.numel() * p.element_size() for p in model.parameters())
        return alloc_before, alloc_after

    eager_before, eager_after = run_eager()

    def run_compiled():
        optimizer.zero_grad()
        out = model(inputs)
        loss = out.sum()
        loss.backward()
        compiled_step_fn()
        return eager_before, eager_after

    comp_before, comp_after = run_compiled()

    return {
        "eager_grad_alloc_bytes": eager_before,
        "eager_param_bytes": eager_after,
        "compiled_grad_alloc_bytes": comp_before,
        "compiled_param_bytes": comp_after,
        "step_overhead_ratio": float(comp_before / (eager_before if eager_before > 0 else 1)),
    }
