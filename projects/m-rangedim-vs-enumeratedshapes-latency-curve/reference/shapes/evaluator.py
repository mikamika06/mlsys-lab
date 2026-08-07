import numpy as np


def evaluate_rangedim(shape_spec, inputs):
    dim_name, min_val, opt_val, max_val = shape_spec
    outputs = []
    for inp in inputs:
        val = int(inp)
        if not (min_val <= val <= max_val):
            val = max(min(val, max_val), min_val)
        out_val = val * 2.5 + 1.0
        outputs.append(out_val)
    return np.array(outputs, dtype=np.float32)


def evaluate_enumerated(shape_list, inputs):
    outputs = []
    sorted_shapes = sorted(shape_list)
    for inp in inputs:
        val = int(inp)
        chosen = sorted_shapes[-1]
        for s in sorted_shapes:
            if s >= val:
                chosen = s
                break
        out_val = val * 2.5 + 1.0
        outputs.append(out_val)
    return np.array(outputs, dtype=np.float32)
