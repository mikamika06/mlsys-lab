import numpy as np

class Node:
    def __init__(self, name, op_type, inputs, outputs):
        self.name = name
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs

class Graph:
    def __init__(self, nodes):
        self.nodes = nodes

    def execute(self, inputs_dict):
        vals = dict(inputs_dict)
        for node in self.nodes:
            if node.op_type == "MatMul":
                vals[node.outputs[0]] = np.matmul(vals[node.inputs[0]], vals[node.inputs[1]])
            elif node.op_type == "Add":
                vals[node.outputs[0]] = vals[node.inputs[0]] + vals[node.inputs[1]]
            elif node.op_type == "Softmax":
                x = vals[node.inputs[0]]
                e = np.exp(x - np.max(x, axis=-1, keepdims=True))
                vals[node.outputs[0]] = e / np.sum(e, axis=-1, keepdims=True)
            elif node.op_type == "FusedAttention":
                q = vals[node.inputs[0]]
                k = vals[node.inputs[1]]
                v = vals[node.inputs[2]]
                scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(q.shape[-1])
                e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                attn = e / np.sum(e, axis=-1, keepdims=True)
                vals[node.outputs[0]] = np.matmul(attn, v)
        return vals[self.nodes[-1].outputs[0]]
