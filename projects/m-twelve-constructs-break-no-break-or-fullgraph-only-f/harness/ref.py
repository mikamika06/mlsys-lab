import torch

CONSTRUCT_TESTS = [
    ("tensor_item", "break"),
    ("shape_print", "break"),
    ("python_if_tensor", "break"),
    ("pure_arithmetic", "no_break"),
    ("tensor_add", "no_break"),
    ("reshape_constant", "no_break"),
    ("fullgraph_only_mutation", "fullgraph_only"),
    ("fullgraph_only_builtin_dict", "fullgraph_only"),
    ("item_in_loop", "break"),
    ("tensor_size_check", "break"),
    ("matrix_multiply", "no_break"),
    ("tensor_clone", "no_break")
]

def classify_constructs(items):
    mapping = dict(CONSTRUCT_TESTS)
    return [mapping.get(x, "break") for x in items]

def evaluate_cond(pred, x, y):
    def true_fn(val_x, val_y):
        return val_x * 2 + val_y
    def false_fn(val_x, val_y):
        return val_x - val_y
    if isinstance(pred, torch.Tensor) and pred.ndim == 0 and pred.item() > 0:
        return true_fn(x, y)
    return false_fn(x, y)

def build_partition_map(log_lines):
    partitions = []
    current = []
    for line in log_lines:
        if "GRAPH BREAK" in line:
            if current:
                partitions.append(current)
                current = []
        else:
            clean = line.strip()
            if clean:
                current.append(clean)
    if current:
        partitions.append(current)
    return partitions
