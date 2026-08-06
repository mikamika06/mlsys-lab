import ref
import numpy as np


def check(workdir):
    from quant.decomposed import naive_int8_matmul, decomposed_matmul
    x, w = ref.generate_data()
    ref_out = np.matmul(x, w)
    naive_out = naive_int8_matmul(x, w)
    decomp_out = decomposed_matmul(x, w, threshold=6.0)

    naive_mse = np.mean((naive_out - ref_out) ** 2)
    decomp_mse = np.mean((decomp_out - ref_out) ** 2)

    out = {"decomposed_better": 1.0 if decomp_mse < naive_mse else 0.0}
    if decomp_mse >= naive_mse:
        out["_note"] = f"decomposed MSE {decomp_mse} not better than naive MSE {naive_mse}"
    return out
