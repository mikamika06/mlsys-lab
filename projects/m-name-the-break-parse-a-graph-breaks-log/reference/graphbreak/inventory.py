import torch


def count_graph_breaks(model, sample_input):
    break_count = 0

    def graph_break_callback(gm, code):
        nonlocal break_count
        break_count += 1
        return gm

    try:
        if isinstance(sample_input, tuple):
            args = sample_input
            kwargs = {}
        elif isinstance(sample_input, dict):
            args = ()
            kwargs = sample_input
        else:
            args = (sample_input,)
            kwargs = {}

        opt_fn = torch.compile(model, backend="eager", fullgraph=False)
        opt_fn(*args, **kwargs)

        compiled_count = 0
        def dummy_compiler(gm, example_inputs):
            nonlocal compiled_count
            compiled_count += 1
            return gm.forward

        opt_count_fn = torch.compile(model, backend=dummy_compiler)
        opt_count_fn(*args, **kwargs)

        break_count = max(0, compiled_count - 1)
    except Exception:
        break_count = 1

    return break_count


def generate_inventory_report(before_model, after_model, sample_input):
    before_breaks = count_graph_breaks(before_model, sample_input)
    after_breaks = count_graph_breaks(after_model, sample_input)
    return {
        "before_breaks": before_breaks,
        "after_breaks": after_breaks,
        "breaks_eliminated": max(0, before_breaks - after_breaks)
    }
