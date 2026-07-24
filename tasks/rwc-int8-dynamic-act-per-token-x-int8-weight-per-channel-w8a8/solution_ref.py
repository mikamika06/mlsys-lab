import numpy as np

def int8_dynamic_act_per_token_x_int8_weight_per_channel(A: np.ndarray,
                                                         W: np.ndarray) -> np.ndarray:
    eps = 1e-12
    # activation scales per row
    scale_row = np.max(np.abs(A), axis=1)
    scale_row[scale_row < eps] = 1.0
    a_int = np.round(A / scale_row[:, None]).clip(-128, 127).astype(np.int8)

    # weight scales per column
    scale_col = np.max(np.abs(W), axis=0)
    scale_col[scale_col < eps] = 1.0
    w_int = np.round(W / scale_col[None, :]).clip(-128, 127).astype(np.int8)

    y_int32 = a_int.astype(np.int32) @ w_int.astype(np.int32)
    Y = y_int32.astype(np.float64) * (scale_row[:, None] * scale_col[None, :])
    return Y
