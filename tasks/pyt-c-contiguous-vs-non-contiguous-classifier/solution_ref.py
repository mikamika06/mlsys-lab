def predict_c_contiguous(ops: list[str]) -> list[bool]:
    """Track shape, strides, and contiguity status of a 2D array subjected to strided operations."""
    # Base array is shape (4, 6) with canonical C strides (6, 1)
    shape = [4, 6]
    strides = [6, 1]
    is_contiguous = True

    answers = []
    for op in ops:
        if op == "transpose":
            if len(shape) == 2:
                shape = [shape[1], shape[0]]
                strides = [strides[1], strides[0]]
            # Contiguity depends on whether strides match canonical layout
            is_contiguous = len(shape) == 1 or strides == [shape[1], 1]
        elif op == "slice_step2":
            shape[0] = (shape[0] + 1) // 2
            strides[0] = strides[0] * 2
            is_contiguous = len(shape) == 1 or strides == [shape[1], 1]
        elif op == "reshape_flat":
            total_elements = 1
            for s in shape:
                total_elements *= s
            shape = [total_elements]
            strides = [1]
            is_contiguous = True
        elif op == "flip":
            strides[0] = -strides[0]
            is_contiguous = False
        else:
            raise ValueError(op)

        answers.append(is_contiguous)

    return answers
