import torch
import torch.export
from export_verify.verifier import verify_graph_signature


class SampleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.randn(4, 4))
        self.register_buffer("b", torch.randn(4, 4))

    def forward(self, x):
        return x @ self.p + self.b


def test_verify_graph_signature_catches_mismatch():
    mod = SampleModule()
    x = torch.randn(4, 4)
    ep = torch.export.export(mod, (x,))

    valid, details = verify_graph_signature(mod, ep, (x,))
    assert valid is True
    assert details["params_ok"] is True
    assert details["buffers_ok"] is True
    assert details["param_shapes_ok"] is True
    assert details["buffer_shapes_ok"] is True

    mod.p = torch.nn.Parameter(torch.randn(4, 8))
    valid_bad, details_bad = verify_graph_signature(mod, ep, (x,))
    assert valid_bad is False
    assert details_bad["param_shapes_ok"] is False
