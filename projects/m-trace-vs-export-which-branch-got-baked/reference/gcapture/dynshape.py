import torch
from torch.export import Dim


def derive_minimal_dynamic_shapes(model, example_input, failing_inputs):
    dynamic_dims = {}
    for i, inp in enumerate(example_input):
        dim_spec = {}
        for dim_idx, s in enumerate(inp.shape):
            varying_sizes = {f_inp[i].shape[dim_idx] for f_inp in failing_inputs}
            varying_sizes.add(s)
            if len(varying_sizes) > 1:
                min_s = min(varying_sizes)
                max_s = max(varying_sizes)
                dim_spec[dim_idx] = Dim(
                    f"dim_{i}_{dim_idx}", min=min_s, max=max_s
                )
        if dim_spec:
            dynamic_dims[i] = dim_spec
        else:
            dynamic_dims[i] = None

    if isinstance(example_input, tuple):
        return tuple(dynamic_dims[i] for i in range(len(example_input)))
    return dynamic_dims[0]
