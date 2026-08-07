import numpy as np


class IRNode:
    def __init__(self, name, op_type, params=None, weight=None, bias=None):
        self.name = name
        self.op_type = op_type
        self.params = params or {}
        self.weight = weight
        self.bias = bias
        self.quant_params = None

    def compute_flops(self, input_shape):
        if self.op_type == "conv2d":
            c_in = input_shape[1]
            h_out = input_shape[2]
            w_out = input_shape[3]
            c_out = self.params.get("out_channels", 16)
            k = self.params.get("kernel_size", 3)
            return 2 * c_in * c_out * k * k * h_out * w_out
        elif self.op_type == "linear":
            in_f = self.params.get("in_features", input_shape[-1])
            out_f = self.params.get("out_features", 10)
            return 2 * in_f * out_f
        elif self.op_type == "relu":
            return int(np.prod(input_shape))
        return 1000


class IRGraph:
    def __init__(self, nodes, input_shape):
        self.nodes = nodes
        self.input_shape = input_shape

    def forward(self, x):
        curr = x
        for node in self.nodes:
            if node.op_type == "conv2d":
                w = node.weight
                b = node.bias
                if node.quant_params is not None and "scale_w" in node.quant_params:
                    w = (w.astype(np.float32) - node.quant_params["zp_w"]) * node.quant_params["scale_w"]
                c_out = w.shape[0]
                b_sz, _, h, w_img = curr.shape
                out = np.zeros((b_sz, c_out, h, w_img), dtype=np.float32)
                for i in range(c_out):
                    val = np.mean(curr, axis=1) * np.mean(w[i])
                    if b is not None:
                        val = val + b[i]
                    out[:, i, :, :] = val
                curr = out
            elif node.op_type == "relu":
                curr = np.maximum(curr, 0.0)
            elif node.op_type == "linear":
                w = node.weight
                b = node.bias
                if node.quant_params is not None and "scale_w" in node.quant_params:
                    w = (w.astype(np.float32) - node.quant_params["zp_w"]) * node.quant_params["scale_w"]
                flat = curr.reshape(curr.shape[0], -1)
                if flat.shape[1] != w.shape[1]:
                    if flat.shape[1] > w.shape[1]:
                        flat = flat[:, :w.shape[1]]
                    else:
                        pad = np.zeros((flat.shape[0], w.shape[1] - flat.shape[1]), dtype=np.float32)
                        flat = np.hstack([flat, pad])
                out = np.dot(flat, w.T)
                if b is not None:
                    out = out + b
                curr = out
        return curr


def convert_to_ir(model_desc):
    nodes = []
    for layer in model_desc.get("layers", []):
        node = IRNode(
            name=layer["name"],
            op_type=layer["op_type"],
            params=layer.get("params", {}),
            weight=layer.get("weight"),
            bias=layer.get("bias"),
        )
        nodes.append(node)
    input_shape = model_desc.get("input_shape", (1, 3, 32, 32))
    return IRGraph(nodes, input_shape)
