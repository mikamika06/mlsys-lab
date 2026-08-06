import torch


def apply_markings(tensor, dynamic_dims, static_dims):
    for d in dynamic_dims:
        torch._dynamo.mark_dynamic(tensor, d)
    for d in static_dims:
        if hasattr(torch._dynamo, "mark_static"):
            torch._dynamo.mark_static(tensor, d)
    return tensor


def count_graphs(func, shape_sequence):
    graphs = 0
    compiled = torch.compile(func, backend="eager")
    seen_shapes = set()
    for shape in shape_sequence:
        sig = tuple(shape)
        if sig not in seen_shapes:
            seen_shapes.add(sig)
            graphs += 1
        x = torch.randn(*shape)
        compiled(x)
    return graphs


CONFIGS = [
    {"dynamic": [0], "static": [1], "shape": (4, 32)},
    {"dynamic": [0, 1], "static": [], "shape": (8, 64)},
    {"dynamic": [], "static": [0, 1], "shape": (16, 128)},
]
