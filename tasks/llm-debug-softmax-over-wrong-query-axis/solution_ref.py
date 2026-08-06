import math

def sdpa(query: list[list[list[float]]],
         key: list[list[list[float]]],
         value: list[list[list[float]]],
         scale: float | None = None) -> list[list[list[float]]]:
    """
    Scaled dot‑product attention with correct softmax axis.

    Parameters
    ----------
    query : list[list[list[float]]]
        Shape (B, N_q, d_k)
    key : list[list[list[float]]]
        Shape (B, N_k, d_k)
    value : list[list[list[float]]]
        Shape (B, N_k, d_v)
    scale : float | None, optional
        Scaling factor. If None, defaults to 1/sqrt(d_k).

    Returns
    -------
    list[list[list[float]]]
        Attention output of shape (B, N_q, d_v).
    """
    if not isinstance(query, list) or not isinstance(key, list) or not isinstance(value, list):
        raise ValueError("All inputs must be 3‑D lists.")

    B = len(query)
    if B == 0 or len(key) != B or len(value) != B:
        raise ValueError("Incompatible batch size.")

    for b in range(B):
        if not isinstance(query[b], list) or not isinstance(key[b], list) or not isinstance(value[b], list):
            raise ValueError("All inputs must be 3‑D lists.")

    Nq = len(query[0])
    Nk = len(key[0])
    if Nq == 0 or Nk == 0 or len(value[0]) != Nk:
        raise ValueError("Incompatible key/value dimensions.")

    dk = len(query[0][0])
    if dk == 0 or len(key[0][0]) != dk:
        raise ValueError("Incompatible query/key dimension d_k.")

    dv = len(value[0][0])

    for b in range(B):
        if len(query[b]) != Nq or len(key[b]) != Nk or len(value[b]) != Nk:
            raise ValueError("Inconsistent sequence lengths.")
        for i in range(Nq):
            if len(query[b][i]) != dk:
                raise ValueError("Inconsistent d_k in query.")
        for j in range(Nk):
            if len(key[b][j]) != dk:
                raise ValueError("Inconsistent d_k in key.")
            if len(value[b][j]) != dv:
                raise ValueError("Inconsistent d_v in value.")

    if scale is None:
        scale = 1.0 / math.sqrt(dk)

    output = []
    for b in range(B):
        batch_out = []
        for i in range(Nq):
            # Compute dot products with all keys for query i
            scores = []
            for j in range(Nk):
                dot_prod = 0.0
                for k in range(dk):
                    dot_prod += query[b][i][k] * key[b][j][k]
                scores.append(dot_prod * scale)

            # Softmax over key dimension
            max_score = max(scores)
            exp_scores = [math.exp(s - max_score) for s in scores]
            sum_exp = sum(exp_scores)
            probs = [exp / sum_exp for exp in exp_scores]

            # Weighted sum over values
            row_out = []
            for d in range(dv):
                val = 0.0
                for j in range(Nk):
                    val += probs[j] * value[b][j][d]
                row_out.append(val)
            batch_out.append(row_out)
        output.append(batch_out)

    return output
