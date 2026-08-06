import numpy as np
from shapecheck.constraints import confirm_derived_dimension


def verify_export_shape(shape, chain, constraints):
    if not confirm_derived_dimension(constraints):
        return False
    current = list(shape)
    for op in chain:
        if op["type"] == "reshape":
            current = op["target"]
        elif op["type"] == "flatten":
            current = [np.prod(current)]
    return len(current) > 0
