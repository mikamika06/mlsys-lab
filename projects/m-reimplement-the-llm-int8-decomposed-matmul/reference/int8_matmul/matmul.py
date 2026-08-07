import numpy as np
from int8_matmul.dequant import derive_vector_scales


def decomposed_matmul(x, w, threshold):
    abs_x = np.abs(x)
    outlier_mask = np.max(abs_x, axis=0) > threshold if x.ndim == 2 else abs_x > threshold
    if x.ndim == 2 and outlier_mask.ndim == 1:
        outlier_indices = np.where(outlier_mask)[0]
        normal_indices = np.where(~outlier_mask)[0]

        if len(outlier_indices) > 0:
            x_out = x[:, outlier_indices]
            w_out = w[outlier_indices, :]
            part_out = np.matmul(x_out, w_out)
        else:
            part_out = 0.0

        if len(normal_indices) > 0:
            x_norm = x[:, normal_indices]
            w_norm = w[normal_indices, :]

            x_scales = derive_vector_scales(x_norm)
            w_scales = derive_vector_scales(w_norm.T).T

            x_int8 = np.clip(np.round(x_norm / x_scales), -128, 127).astype(np.int8)
            w_int8 = np.clip(np.round(w_norm / w_scales), -128, 127).astype(np.int8)

            matmul_int8 = np.matmul(x_int8.astype(np.float32), w_int8.astype(np.float32))
            part_norm = matmul_int8 * x_scales * w_scales
        else:
            part_norm = 0.0

        return part_out + part_norm
    else:
        return np.matmul(x, w)
