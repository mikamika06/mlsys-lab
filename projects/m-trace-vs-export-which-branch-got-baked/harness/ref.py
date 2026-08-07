import torch
from torch.export import Dim


class ConditionalModule(torch.nn.Module):

    def forward(self, x):
        if x.mean() > 0.0:
            return x * 2.0 + 1.0
        return x * 0.5 - 3.0


class DynamicShapeModule(torch.nn.Module):

    def forward(self, x, y):
        return torch.matmul(x, y)


def get_branch_test_setup():
    mod = ConditionalModule()
    inp1 = torch.tensor([1.0, 3.0, 5.0])
    inp2 = torch.tensor([-2.0, -4.0, -1.0])
    return mod, inp1, inp2


def get_dynshape_test_setup():
    mod = DynamicShapeModule()
    example_input = (torch.randn(4, 8), torch.randn(8, 16))
    failing_inputs = [
        (torch.randn(2, 8), torch.randn(8, 16)),
        (torch.randn(7, 8), torch.randn(8, 16)),
    ]
    return mod, example_input, failing_inputs


def ref_inspect_baked_branch(model, example_input, alternate_input):
    traced = torch.jit.trace(model, example_input)
    out_alt_traced = traced(alternate_input)
    out_alt_eager = model(alternate_input)

    trace_baked = not torch.allclose(out_alt_traced, out_alt_eager, atol=1e-4)
    example_mean = float(example_input.mean().item())
    branch_name = "then_branch" if example_mean > 0.0 else "else_branch"

    export_ok = True
    try:
        ep = torch.export.export(model, (alternate_input,))
        ep.module()(alternate_input)
    except Exception:
        export_ok = False

    return {
        "trace_baked_branch": trace_baked,
        "trace_took_branch": branch_name,
        "export_supports_alt": export_ok,
    }


def ref_derive_minimal_dynamic_shapes(model, example_input, failing_inputs):
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


def ref_aten_op_histogram(exported_program):
    counts = {}
    graph_module = exported_program.graph_module
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            target = node.target
            name = getattr(target, "__name__", str(target))
            counts[name] = counts.get(name, 0) + 1
    return counts
