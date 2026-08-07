import math


def normalized_hadamard(n: int) -> list[list[float]]:
    h = [[1.0]]
    while len(h) < n:
        new_h = []
        for row in h:
            new_h.append(row + row)
        for row in h:
            neg_row = []
            for val in row:
                neg_row.append(-val)
            new_h.append(row + neg_row)
        h = new_h

    scale = math.sqrt(n)
    result = []
    for i in range(n):
        row_result = []
        for j in range(n):
            row_result.append(h[i][j] / scale)
        result.append(row_result)

    return result
