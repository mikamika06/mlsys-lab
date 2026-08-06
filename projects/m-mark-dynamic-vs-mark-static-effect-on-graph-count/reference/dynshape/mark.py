import torch


def apply_markings(tensor, dynamic_dims, static_dims):
    for d in dynamic_dims:
        torch._dynamo.mark_dynamic(tensor, d)
    for d in static_dims:
        if hasattr(torch._dynamo, "mark_static"):
            torch._dynamo.mark_static(tensor, d)
    return tensor
