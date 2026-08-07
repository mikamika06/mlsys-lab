def streamed_mlp_forward(layers, x):
    out = [row[:] for row in x]
    for i, layer in enumerate(layers):
        w = layer["w"]
        b = layer["b"]

        nrows = len(out)
        ncols = len(w[0])
        w_rows = len(w)

        new_out = [[0.0 for _ in range(ncols)] for _ in range(nrows)]
        for r in range(nrows):
            for c in range(ncols):
                val = 0.0
                for k in range(w_rows):
                    val += out[r][k] * w[k][c]
                val += b[c]
                new_out[r][c] = val
        out = new_out

        del w
        del b
        if i != len(layers) - 1:
            nrows_m = len(out)
            ncols_m = len(out[0])
            new_out_relu = [[0.0 for _ in range(ncols_m)] for _ in range(nrows_m)]
            for r in range(nrows_m):
                for c in range(ncols_m):
                    val = out[r][c]
                    if val < 0.0:
                        val = 0.0
                    new_out_relu[r][c] = val
            out = new_out_relu
    return out
