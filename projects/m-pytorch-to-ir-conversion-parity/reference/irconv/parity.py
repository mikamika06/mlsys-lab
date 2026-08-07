import numpy as np


def compute_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a) + 1e-12
    return float(num / den)


def evaluate_ir_node(op_type: str, inputs: list[np.ndarray], attributes: dict) -> np.ndarray:
    if op_type == "Add":
        return inputs[0] + inputs[1]
    elif op_type == "MatMul":
        return np.matmul(inputs[0], inputs[1])
    elif op_type == "ReLU":
        return np.maximum(inputs[0], 0)
    elif op_type == "Reshape":
        new_shape = attributes.get("shape")
        return np.reshape(inputs[0], new_shape)
    else:
        raise ValueError(f"Unsupported operation: {op_type}")


def verify_conversion_parity(pytorch_outputs: dict[str, np.ndarray], ir_graph: list[dict], tol: float = 1e-4) -> dict[str, float]:
    computed_outputs = {}
    errors = {}
    for node in ir_graph:
        node_inputs = []
        for inp_name in node["inputs"]:
            if inp_name in computed_outputs:
                node_inputs.append(computed_outputs[inp_name])
            elif inp_name in pytorch_outputs:
                node_inputs.append(pytorch_outputs[inp_name])
            else:
                raise KeyError(f"Input tensor {inp_name} not found")
        out = evaluate_ir_node(node["op"], node_inputs, node.get("attributes", {}))
        name = node["output"]
        computed_outputs[name] = out
        if name in pytorch_outputs:
            err = compute_relative_error(pytorch_outputs[name], out)
            errors[name] = err
    return errors
