class LazyNode:
    def __init__(self, op, inputs, val=None):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

def build_lazy_lora_graph(x, w_base, lora_a, lora_b, scale=1.0):
    raise NotImplementedError

def measure_execution(root_node, force_eval=False):
    raise NotImplementedError
