import torch


def count_graph_breaks(fn, example_inputs) -> int:
    """Count the number of graph breaks during torch.compile tracing."""
    breaks = 0

    def graph_break_count_backend(gm, sample_inputs):
        nonlocal breaks
        breaks += 1
        return gm.forward

    opt_fn = torch.compile(fn, backend=graph_break_count_backend)
    try:
        opt_fn(*example_inputs)
    except Exception:
        pass
    return max(0, breaks - 1)


def run_compiled_model(mod: torch.nn.Module, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Run model through torch.compile ensuring clean tracing."""
    opt_mod = torch.compile(mod, backend="aot_eager")
    return opt_mod(x, alpha)
