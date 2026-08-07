import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from coreml_exporter.converter import export_and_verify
from coreml_exporter.precision import compare_precisions


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


def test_export_numerical_accuracy():
    torch.manual_seed(42)
    model = SimpleNet()
    example_input = (torch.randn(1, 10),)
    eval_input = (torch.randn(1, 10),)

    _, err = export_and_verify(model, example_input, eval_input, "dummy.mlpackage")
    assert err <= 1e-4, f"Max absolute error {err} exceeds threshold 1e-4"


def test_precision_compression_ratio():
    torch.manual_seed(42)
    model = SimpleNet()
    example_input = (torch.randn(1, 10),)
    eval_input = (torch.randn(1, 10),)

    metrics = compare_precisions(model, example_input, eval_input, "/tmp")
    assert metrics["ratio"] <= 0.6, f"Compression ratio {metrics['ratio']} is > 0.6"
