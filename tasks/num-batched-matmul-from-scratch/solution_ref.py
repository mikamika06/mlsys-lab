def batched_matmul(A: list, B: list) -> list:
    """
    Performs batched matrix multiplication without using any external libraries.
    A: 3D list of shape (batch_size, n, m)
    B: 3D list of shape (batch_size, m, p)
    Returns a 3D list of shape (batch_size, n, p)
    """
    result = []
    for a_mat, b_mat in zip(A, B):
        # Transpose B for easier column access
        b_t = list(zip(*b_mat))
        c_mat = [
            [sum(a * b for a, b in zip(a_row, b_col)) for b_col in b_t]
            for a_row in a_mat
        ]
        result.append(c_mat)
    return result
