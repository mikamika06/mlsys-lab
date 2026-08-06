import numpy as np


def slice_quantized_matrix(w_q, scales, g_idx, tp_size, mode="row"):
    k, n = w_q.shape
    if mode == "row":
        if k % tp_size != 0:
            raise ValueError("k not divisible by tp_size")
        k_rank = k // tp_size
        results = []
        for r in range(tp_size):
            w_slice = w_q[r * k_rank : (r + 1) * k_rank, :]
            g_slice = g_idx[r * k_rank : (r + 1) * k_rank]
            unique_g = np.unique(g_slice)
            if len(unique_g) == 0:
                results.append({
                    "w_q_rank": w_slice,
                    "scales_rank": np.empty((0, n)),
                    "g_idx_rank": g_slice,
                    "valid": True,
                    "error": None,
                })
                continue
            min_g, max_g = int(np.min(unique_g)), int(np.max(unique_g))
            is_contig = (max_g - min_g + 1 == len(unique_g)) and np.array_equal(
                np.sort(unique_g), np.arange(min_g, max_g + 1)
            )

            if is_contig and min_g >= 0 and max_g < scales.shape[0]:
                s_slice = scales[min_g : max_g + 1, :]
                g_remapped = g_slice - min_g
                results.append({
                    "w_q_rank": w_slice,
                    "scales_rank": s_slice,
                    "g_idx_rank": g_remapped,
                    "valid": True,
                    "error": None,
                })
            else:
                results.append({
                    "w_q_rank": w_slice,
                    "scales_rank": np.empty((0, n)),
                    "g_idx_rank": g_slice,
                    "valid": False,
                    "error": "out_of_bounds_group_access",
                })
        return results
    else:
        if n % tp_size != 0:
            raise ValueError("n not divisible by tp_size")
        n_rank = n // tp_size
        results = []
        for r in range(tp_size):
            w_slice = w_q[:, r * n_rank : (r + 1) * n_rank]
            s_slice = scales[:, r * n_rank : (r + 1) * n_rank]
            results.append({
                "w_q_rank": w_slice,
                "scales_rank": s_slice,
                "g_idx_rank": g_idx,
                "valid": True,
                "error": None,
            })
        return results
