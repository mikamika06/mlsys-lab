def classify_contiguity(arr: list[list[float]]) -> str:
    """
    Return "C", "F" or "Neither" based on row-major vs column-major order.
    """
    rows = len(arr)
    if rows <= 1:
        return "Neither"
    cols = len(arr[0])
    if cols <= 1:
        return "Neither"

    base = arr[0][0]

    is_c = True
    for i in range(rows):
        for j in range(cols):
            if arr[i][j] - base != i * cols + j:
                is_c = False
                break
        if not is_c:
            break

    is_f = True
    for i in range(rows):
        for j in range(cols):
            if arr[i][j] - base != j * rows + i:
                is_f = False
                break
        if not is_f:
            break

    if is_c and not is_f:
        return "C"
    elif is_f and not is_c:
        return "F"
    else:
        return "Neither"
