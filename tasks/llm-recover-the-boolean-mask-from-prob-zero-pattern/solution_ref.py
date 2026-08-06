def recover_mask(P: list[list[float]], eps: float = 1e-12) -> list[list[bool]]:
    res = []
    for row in P:
        row_res = []
        for val in row:
            row_res.append(val > eps)
        res.append(row_res)
    return res
