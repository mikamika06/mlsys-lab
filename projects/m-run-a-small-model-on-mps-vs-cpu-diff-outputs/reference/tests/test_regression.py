import torch
from mpscheck.compare import compare_outputs
from mpscheck.cost import measure_staging_cost
from mpscheck.unsupported import extract_unsupported_op

def test_compare_outputs_basic():
    model = torch.nn.Linear(10, 5)
    x = torch.randn(2, 10)
    err = compare_outputs(model, x)
    assert isinstance(err, float)
    assert err >= 0.0

def test_measure_staging_cost_keys():
    sizes = [64, 128]
    costs = measure_staging_cost(sizes)
    assert all(s in costs for s in sizes)

def test_extract_unsupported_op_catch():
    def bad_func():
        raise RuntimeError("The operator 'aten::foo' is not implemented for the MPS device")
    op = extract_unsupported_op(bad_func)
    assert op == "aten::foo"
