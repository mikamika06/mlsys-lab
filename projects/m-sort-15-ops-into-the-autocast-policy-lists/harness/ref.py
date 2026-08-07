OPS_LIST = [
    "aten.mm", "aten.bmm", "aten.addmm", "aten.conv2d", "aten.linear",
    "aten.sum", "aten.mean", "aten.softmax", "aten.layer_norm", "aten.gelu",
    "aten.pow", "aten.div", "aten.exp", "aten.log", "aten.sin"
]

POLICY_MAP = {
    "aten.mm": "fp16",
    "aten.bmm": "fp16",
    "aten.addmm": "fp16",
    "aten.conv2d": "fp16",
    "aten.linear": "fp16",
    "aten.sum": "promote",
    "aten.mean": "promote",
    "aten.softmax": "promote",
    "aten.layer_norm": "promote",
    "aten.gelu": "promote",
    "aten.pow": "promote",
    "aten.div": "promote",
    "aten.exp": "promote",
    "aten.log": "promote",
    "aten.sin": "promote"
}

def classify_ops():
    return POLICY_MAP

def custom_op_forward(x, weight):
    return x @ weight

def custom_op_backward(grad_output, x, weight):
    return grad_output @ weight.t(), x.t() @ grad_output
