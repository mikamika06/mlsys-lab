import math

def ffn_forward(
    x: list[float],
    W_up: list[list[float]],
    b_up: list[float],
    W_down: list[list[float]],
    b_down: list[float],
) -> list[float]:
    """Vanilla FFN forward: down_proj(gelu(up_proj(x)))."""
    d_hidden = len(W_up)
    d = len(W_down)

    h = [0.0] * d_hidden
    for i in range(d_hidden):
        acc = b_up[i]
        for j in range(len(x)):
            acc += W_up[i][j] * x[j]
        h[i] = acc

    a = [0.0] * d_hidden
    const = math.sqrt(2.0 / math.pi)
    for i in range(d_hidden):
        val = h[i]
        gelu_val = 0.5 * val * (1.0 + math.tanh(const * (val + 0.044715 * (val ** 3))))
        a[i] = gelu_val

    res = [0.0] * d
    for i in range(d):
        acc = b_down[i]
        for j in range(d_hidden):
            acc += W_down[i][j] * a[j]
        res[i] = acc

    return res
