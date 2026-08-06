from ablation.parser import parse_graph_code

def test_parser_regression():
    code = """
def forward(self, arg0, arg1):
    add = torch.ops.aten.add.Tensor(arg0, arg1)
    sub = torch.ops.aten.sub.Tensor(add, arg1)
    return (sub,)
"""
    ops = parse_graph_code(code)
    assert len(ops) == 2
    assert ops[0] == "torch.ops.aten.add.Tensor"
    assert ops[1] == "torch.ops.aten.sub.Tensor"
