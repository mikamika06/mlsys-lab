import torch


def count_graph_breaks(fn, example_inputs) -> int:
    """Count the number of graph breaks during torch.compile tracing."""
    raise NotImplementedError


def run_compiled_model(mod: torch.nn.Module, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Run model through torch.compile ensuring clean tracing."""
    raise NotImplementedError
