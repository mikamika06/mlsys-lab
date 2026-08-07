class IRNode:
    def __init__(self, name, op_type, params=None, weight=None, bias=None):
        raise NotImplementedError

    def compute_flops(self, input_shape):
        raise NotImplementedError


class IRGraph:
    def __init__(self, nodes, input_shape):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def convert_to_ir(model_desc):
    raise NotImplementedError
