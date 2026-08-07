import numpy as np


def locate_first_nan(node_outputs):
    for name, val in node_outputs.items():
        arr = np.asarray(val)
        if not np.isfinite(arr).all():
            return name
    return None


def inspect_module_nans(named_tensors):
    offending = {}
    for mod_name, tensor in named_tensors.items():
        arr = np.asarray(tensor)
        if not np.isfinite(arr).all():
            offending[mod_name] = True
        else:
            offending[mod_name] = False
    return offending
