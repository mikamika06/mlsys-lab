import sys
import torch
sys.path.insert(0, ".")
from graphclean.module import CleanModel


def test_model_compiles_fullgraph():
    model = CleanModel()
    x = torch.randn(2, 16)
    compiled = torch.compile(model, fullgraph=True, backend="eager")
    out = compiled(x)
    assert out.shape == (2, 16)


def test_no_python_control_flow_on_tensors():
    import inspect
    source = inspect.getsource(CleanModel.forward)
    assert "if " not in source, "found explicit conditional statement in forward"
    assert "item()" not in source, "found tensor item conversion in forward"
