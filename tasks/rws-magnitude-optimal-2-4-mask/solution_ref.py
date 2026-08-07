def magnitude_optimal_2to4_mask(weights: list[list[float]]) -> list[list[bool]]:
    """
    Return a boolean mask that keeps the two largest‑magnitude weights in each
    consecutive block of four columns.
    """
    if len(weights) == 0 or len(weights[0]) % 4 != 0:
        raise ValueError("last dimension must be a multiple of 4")

    mask = []
    for row in weights:
        row_mask = []
        for i in range(0, len(row), 4):
            block = row[i:i + 4]
            abs_block = [abs(val) for val in block]

            max1_idx = 0
            for j in range(1, 4):
                if abs_block[j] > abs_block[max1_idx]:
                    max1_idx = j

            max2_idx = -1
            for j in range(4):
                if j == max1_idx:
                    continue
                if max2_idx == -1 or abs_block[j] > abs_block[max2_idx]:
                    max2_idx = j

            block_mask = [False, False, False, False]
            block_mask[max1_idx] = True
            block_mask[max2_idx] = True
            row_mask.extend(block_mask)

        mask.append(row_mask)

    return mask
