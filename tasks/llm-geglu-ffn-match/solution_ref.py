import math


def geglu_ffn(x: list[list[float]], w_gate: list[list[float]], w_up: list[list[float]]) -> list[list[float]]:
    """Gated Linear Unit with GELU activation."""
    batch = len(x)
    d_in = len(x[0])
    d_out = len(w_gate[0])

    sqrt_2_pi = math.sqrt(2.0 / math.pi)

    out = []
    for i in range(batch):
        row_out = []
        for j in range(d_out):
            gate_val = 0.0
            up_val = 0.0
            for k in range(d_in):
                gate_val += x[i][k] * w_gate[k][j]
                up_val += x[i][k] * w_up[k][j]

            val = sqrt_2_pi * (gate_val + 0.044715 * gate_val * gate_val * gate_val)
            gelu = 0.5 * gate_val * (1.0 + math.tanh(val))
            row_out.append(gelu * up_val)
        out.append(row_out)

    return out
