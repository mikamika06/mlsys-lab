import torch
from customop.ops import register_custom_op, validate_op_schema
from customop.compile_utils import count_graph_breaks, run_compiled_model


class SampleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha):
        op = register_custom_op()
        res = op(x, alpha)
        return res + 1.0


def test_custom_op_schema_and_compile():
    op = register_custom_op()
    x = torch.randn(4, 4, dtype=torch.float32)
    alpha = 0.5

    assert validate_op_schema(x, alpha) is True

    mod = SampleModule()
    num_breaks = count_graph_breaks(mod, (x, alpha))
    assert num_breaks == 0

    out = run_compiled_model(mod, x, alpha)
    expected = mod(x, alpha)
    assert torch.allclose(out, expected, atol=1e-5)
