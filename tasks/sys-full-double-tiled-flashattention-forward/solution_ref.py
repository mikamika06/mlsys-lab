import numpy as np


def flash_attention_forward(Q, K, V, Br, Bc):
    n = Q.shape[0]
    dv = V.shape[1]
    scale = 1.0 / np.sqrt(Q.shape[1])

    output = np.zeros((n, dv), dtype=np.float64)

    for row_start in range(0, n, Br):
        row_end = min(row_start + Br, n)
        q_tile = Q[row_start:row_end]

        running_max = np.full((row_end - row_start,), -np.inf, dtype=np.float64)
        running_sum = np.zeros((row_end - row_start,), dtype=np.float64)
        running_output = np.zeros((row_end - row_start, dv), dtype=np.float64)

        for col_start in range(0, n, Bc):
            col_end = min(col_start + Bc, n)

            score_tile = (q_tile @ K[col_start:col_end].T) * scale

            tile_max = np.max(score_tile, axis=1)
            new_max = np.maximum(running_max, tile_max)

            old_factor = np.exp(running_max - new_max)
            exp_tile = np.exp(score_tile - new_max[:, None])

            new_sum = old_factor * running_sum + np.sum(exp_tile, axis=1)

            numerator = (
                old_factor[:, None] * running_sum[:, None] * running_output
                + exp_tile @ V[col_start:col_end]
            )
            running_output = numerator / new_sum[:, None]

            running_max = new_max
            running_sum = new_sum

        output[row_start:row_end] = running_output

    return output
