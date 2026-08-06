import torch


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
