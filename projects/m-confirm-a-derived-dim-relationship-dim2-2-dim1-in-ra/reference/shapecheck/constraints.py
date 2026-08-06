def confirm_derived_dimension(constraints):
    dim1 = constraints.get("dim1")
    dim2 = constraints.get("dim2")
    if dim1 is None or dim2 is None:
        return False
    return dim2 == 2 * dim1
