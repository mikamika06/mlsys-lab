"""Tensor classification logic."""

def classify_tensors(graph_spec):
    """Classify graph tensors into execution tensors and shape tensors."""
    execution_tensors = []
    shape_tensors = []
    for tensor in graph_spec.get("inputs", []):
        name = tensor["name"]
        is_shape = tensor.get("is_shape_tensor", False)
        role = tensor.get("role", "execution")
        if is_shape or role == "shape":
            shape_tensors.append(name)
        else:
            execution_tensors.append(name)
    return {
        "execution_tensors": sorted(execution_tensors),
        "shape_tensors": sorted(shape_tensors),
    }
