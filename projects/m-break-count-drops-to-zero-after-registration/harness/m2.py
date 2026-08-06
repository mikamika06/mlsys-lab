import torch
import ref


class BenchmarkModel(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x, alpha):
        h = self.op(x, alpha)
        return torch.sin(h)


def check(workdir):
    out = {"graph_breaks": 1.0, "compiled_correctly": 0.0}
    try:
        from customop.ops import register_custom_op
        from customop.compile_utils import count_graph_breaks, run_compiled_model

        op = register_custom_op()
        x, alpha = ref.make_test_data(123)
        model = BenchmarkModel(op)

        breaks = count_graph_breaks(model, (x, alpha))
        out["graph_breaks"] = float(breaks)

        expected = model(x, alpha)
        actual = run_compiled_model(model, x, alpha)

        if torch.allclose(actual, expected, atol=1e-5):
            out["compiled_correctly"] = 1.0
        else:
            out["_note"] = "Compiled output does not match eager execution."
    except Exception as e:
        out["_note"] = f"Error checking milestone 2: {type(e).__name__}: {str(e)[:120]}"
    return out
