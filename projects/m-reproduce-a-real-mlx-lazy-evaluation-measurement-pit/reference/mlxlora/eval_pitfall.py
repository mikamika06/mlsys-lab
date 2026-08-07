import numpy as np

class LazyNode:
    def __init__(self, op, inputs, val=None):
        self.op = op
        self.inputs = inputs
        self.val = val
        self.evaluated = True if op == "leaf" else False

    def eval(self):
        if self.evaluated:
            return self.val
        in_vals = [inp.eval() if isinstance(inp, LazyNode) else inp for inp in self.inputs]
        if self.op == "add":
            self.val = in_vals[0] + in_vals[1]
        elif self.op == "matmul":
            self.val = in_vals[0] @ in_vals[1]
        elif self.op == "scale":
            self.val = in_vals[0] * in_vals[1]
        self.evaluated = True
        return self.val

def build_lazy_lora_graph(x, w_base, lora_a, lora_b, scale=1.0):
    h_base = LazyNode("matmul", [x, w_base])
    h_a = LazyNode("matmul", [x, lora_a])
    h_b = LazyNode("matmul", [h_a, lora_b])
    h_scaled = LazyNode("scale", [h_b, scale])
    out = LazyNode("add", [h_base, h_scaled])
    return out

def measure_execution(root_node, force_eval=False):
    if force_eval:
        root_node.eval()

    visited = set()
    stack = [root_node]
    evaluated_count = 0

    while stack:
        curr = stack.pop()
        if id(curr) in visited:
            continue
        visited.add(id(curr))
        if isinstance(curr, LazyNode):
            if curr.evaluated:
                evaluated_count += 1
            for inp in curr.inputs:
                if isinstance(inp, LazyNode):
                    stack.append(inp)

    return {
        "evaluated_nodes": evaluated_count,
        "total_nodes": len(visited),
        "computed": root_node.evaluated,
        "result": root_node.val if root_node.evaluated else None
    }
