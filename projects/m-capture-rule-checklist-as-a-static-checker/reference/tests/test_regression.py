import torch
import torch.fx
from graph_checker.checker import check_graph_violations
from graph_checker.optimizer import suggest_safe_transforms

class BadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x.cpu()
        z = torch.empty(x.shape)
        return y + z

def test_static_checker_regression():
    gm = torch.fx.symbolic_trace(BadModel())
    violations = check_graph_violations(gm)
    assert len(violations) >= 2, "Expected violations for cpu transfer and empty allocation"
    
    transformed_gm = suggest_safe_transforms(gm)
    remaining_violations = check_graph_violations(transformed_gm)
    assert len(remaining_violations) == 0, "Transformed model should have no capture violations"
