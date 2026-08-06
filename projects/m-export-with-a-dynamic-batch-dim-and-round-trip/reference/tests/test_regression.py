import sys
import torch
sys.path.insert(0, ".")
from export_util.export import export_with_dynamic_batch
from export_util.signature import reconstruct_signature

class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2.0

def test_export_dynamic_batch():
    model = SimpleModel()
    x = torch.randn(4, 16)
    ep = export_with_dynamic_batch(model, (x,))
    assert ep is not None

def test_signature_reconstruction():
    model = SimpleModel()
    x = torch.randn(2, 8)
    ep = export_with_dynamic_batch(model, (x,))
    sig = reconstruct_signature(ep)
    assert len(sig) > 0
