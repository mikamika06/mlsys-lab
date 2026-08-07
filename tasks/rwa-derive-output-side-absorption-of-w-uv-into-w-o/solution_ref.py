def absorb_w_uv(W_O: list[list[float]], W_UV: list[list[float]], P: list[list[float]], c_V: list[list[float]]) -> list[list[float]]:
    rows_p = len(P)
    cols_p = len(P[0]) if rows_p > 0 else 0
    cols_cv = len(c_V[0]) if len(c_V) > 0 else 0

    latent_attention = [[0.0 for _ in range(cols_cv)] for _ in range(rows_p)]
    for i in range(rows_p):
        for k in range(cols_p):
            pik = P[i][k]
            for j in range(cols_cv):
                latent_attention[i][j] += pik * c_V[k][j]

    rows_wo = len(W_O)
    cols_wo = len(W_O[0]) if rows_wo > 0 else 0
    cols_wuv = len(W_UV[0]) if len(W_UV) > 0 else 0

    absorbed_output = [[0.0 for _ in range(cols_wuv)] for _ in range(rows_wo)]
    for i in range(rows_wo):
        for k in range(cols_wo):
            woik = W_O[i][k]
            for j in range(cols_wuv):
                absorbed_output[i][j] += woik * W_UV[k][j]

    absorbed_output_t = [[0.0 for _ in range(rows_wo)] for _ in range(cols_wuv)]
    for i in range(rows_wo):
        for j in range(cols_wuv):
            absorbed_output_t[j][i] = absorbed_output[i][j]

    rows_la = len(latent_attention)
    cols_la = len(latent_attention[0]) if rows_la > 0 else 0
    cols_aot = len(absorbed_output_t[0]) if len(absorbed_output_t) > 0 else 0

    result = [[0.0 for _ in range(cols_aot)] for _ in range(rows_la)]
    for i in range(rows_la):
        for k in range(cols_la):
            laik = latent_attention[i][k]
            for j in range(cols_aot):
                result[i][j] += laik * absorbed_output_t[k][j]

    return result
