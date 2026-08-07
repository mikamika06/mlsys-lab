def pick_kivi_quant_axis(K: list[list[float]], V: list[list[float]]) -> tuple[str, str]:
    """
    For each of K and V choose the axis (channel or token) that yields a lower
    group‑quantisation mean squared error.  The MSE is computed by summing the
    population variances over groups.
    """
    def _label(arr: list[list[float]]) -> str:
        rows = len(arr)
        cols = len(arr[0])

        channel_var = 0.0
        for c in range(cols):
            mean_c = 0.0
            for r in range(rows):
                mean_c += arr[r][c]
            mean_c /= rows

            var_c = 0.0
            for r in range(rows):
                diff = arr[r][c] - mean_c
                var_c += diff * diff
            var_c /= rows
            channel_var += var_c

        token_var = 0.0
        for r in range(rows):
            mean_r = 0.0
            for c in range(cols):
                mean_r += arr[r][c]
            mean_r /= cols

            var_r = 0.0
            for c in range(cols):
                diff = arr[r][c] - mean_r
                var_r += diff * diff
            var_r /= cols
            token_var += var_r

        return "channel" if channel_var <= token_var else "token"

    k_axis = _label(K)
    v_axis = _label(V)
    return k_axis, v_axis
