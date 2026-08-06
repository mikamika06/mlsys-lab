class MILOp:
    def __init__(self, name, inputs, outputs, attributes=None):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or {}

    def to_dict(self):
        return {
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes
        }

def build_three_op_program():
    op1 = MILOp("transpose", ["x"], ["x_t"], {"perm": [0, 2, 1, 3]})
    op2 = MILOp("matmul", ["x_t", "weights"], ["attn_score"], {})
    op3 = MILOp("softmax", ["attn_score"], ["attn_out"], {"axis": -1})
    return [op1.to_dict(), op2.to_dict(), op3.to_dict()]
