import math

def _silu(x: float) -> float:
    """Sigmoid‑linear unit."""
    return x / (1.0 + math.exp(-x))

def swiglu(
    X: list[list[float]],
    W_gate: list[list[float]],
    W_up: list[list[float]],
    b_gate: list[float] | None = None,
    b_up: list[float] | None = None
) -> list[list[float]]:
    """
    Compute the SwiGLU activation using plain Python lists and loops.

    Parameters
    ----------
    X : (n, d_in) list of inputs.
    W_gate, W_up : weight matrices of shape (d_in, d_out).
    b_gate, b_up : optional bias vectors of length d_out; treated as zero if None.

    Returns
    -------
    Y : (n, d_out) list of SwiGLU outputs.
    """
    n = len(X)
    d_in = len(X[0])
    d_out = len(W_gate[0])

    gate = []
    for i in range(n):
        row = []
        for j in range(d_out):
            val = 0.0
            for k in range(d_in):
                val += X[i][k] * W_gate[k][j]
            if b_gate is not None:
                val += b_gate[j]
            row.append(val)
        gate.append(row)

    up = []
    for i in range(n):
        row = []
        for j in range(d_out):
            val = 0.0
            for k in range(d_in):
                val += X[i][k] * W_up[k][j]
            if b_up is not None:
                val += b_up[j]
            row.append(val)
        up.append(row)

    Y = []
    for i in range(n):
        row = []
        for j in range(d_out):
            g_val = gate[i][j]
            u_val = _silu(up[i][j])
            row.append(g_val * u_val)
        Y.append(row)

    return Y
