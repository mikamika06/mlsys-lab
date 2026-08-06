import numpy as np


def streamed_mlp_forward(layers, x):
    out = np.asarray(x, dtype=np.float64)
    for i, layer in enumerate(layers):
        w = layer["w"]
        b = layer["b"]
        
        nrows = out.shape[0]
        ncols = w.shape[1]
        w_rows = w.shape[0]
        
        new_out = np.zeros((nrows, ncols), dtype=np.float64)
        for r in range(nrows):
            for c in range(ncols):
                val = 0.0
                for k in range(w_rows):
                    val += out[r, k] * w[k, c]
                val += b[c]
                new_out[r, c] = val
        out = new_out

        del w
        del b
        if i != len(layers) - 1:
            nrows_m = out.shape[0]
            ncols_m = out.shape[1]
            new_out_relu = np.zeros((nrows_m, ncols_m), dtype=np.float64)
            for r in range(nrows_m):
                for c in range(ncols_m):
                    val = out[r, c]
                    if val < 0.0:
                        val = 0.0
                    new_out_relu[r, c] = val
            out = new_out_relu
    return out
