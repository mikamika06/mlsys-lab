import torch


def inspect_baked_branch(model, example_input, alternate_input):
    traced = torch.jit.trace(model, example_input)
    out_orig = traced(example_input)
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
