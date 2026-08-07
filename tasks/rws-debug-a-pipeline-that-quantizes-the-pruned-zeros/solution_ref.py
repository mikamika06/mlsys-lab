def compound_prune_quantize_2_4(W: list[list[float]], nbits: int = 4) -> list[list[float]]:
    """Compound 2:4 structured pruning + per-group int quantization.

    `W` is a 2-D list of floats whose last dimension is a multiple of 4. Every
    consecutive block of 4 elements along the last axis is treated as one
    N:M block *and* one quantization group:

    1. Structured 2:4 prune: zero the 2 smallest-magnitude elements of the
       block, keep the 2 largest (the "survivors").
    2. The block's quantization scale is the mean magnitude of its
       survivors only:

           scale = mean(|survivors|)

       (if a block had no survivors to begin with -- i.e. it was already
       all zero -- use `scale = 1.0`).
    3. Each survivor `v` is quantized/dequantized in place:
       `code = clip(round(v / scale), -qmax, qmax)`, `dequant = code *
       scale`, with `qmax = 2 ** (nbits - 1) - 1`. Pruned positions stay
       exactly `0.0` -- they need no code at all.

    Parameters
    ----------
    W : list[list[float]]
    nbits : int

    Returns
    -------
    W_hat : list[list[float]]
    """
    qmax = 2 ** (nbits - 1) - 1
    out_rows = []

    for row in W:
        new_row = []
        for i in range(0, len(row), 4):
            block = row[i:i+4]
            indexed_abs = [(abs(val), idx, val) for idx, val in enumerate(block)]
            # Sort by absolute value ascending, then index ascending for stability
            indexed_abs.sort(key=lambda x: (x[0], x[1]))

            # The 2 largest are the last two in the sorted list
            survivor_indices = {item[1] for item in indexed_abs[2:]}

            survivor_abs_sum = 0.0
            survivor_count = 0
            for idx, val in enumerate(block):
                if idx in survivor_indices:
                    survivor_abs_sum += abs(val)
                    survivor_count += 1

            if survivor_count > 0:
                scale = survivor_abs_sum / survivor_count
            else:
                scale = 1.0

            new_block = []
            for idx, val in enumerate(block):
                if idx in survivor_indices:
                    code = max(-qmax, min(qmax, round(val / scale)))
                    new_block.append(code * scale)
                else:
                    new_block.append(0.0)
            new_row.extend(new_block)
        out_rows.append(new_row)

    return out_rows
