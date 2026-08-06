import math

def apply_rope(x: list[list[list[list[float]]]], pos: list[int]) -> list[list[list[list[float]]]]:
    """Apply Rotary Position Embedding (RoPE) to the last dimension of x using pure Python."""
    B = len(x)
    S = len(x[0])
    H = len(x[0][0])
    D = len(x[0][0][0])

    half_D = D // 2
    freqs = [1.0 / (10000.0 ** ((2 * i) / D)) for i in range(half_D)]

    out = []
    for b in range(B):
        batch_out = []
        for s in range(S):
            m = float(pos[s])
            seq_out = []
            for h in range(H):
                head_data = x[b][s][h]
                new_head_data = [0.0] * D
                for i in range(half_D):
                    angle = m * freqs[i]
                    c = math.cos(angle)
                    sin_val = math.sin(angle)

                    x_even = head_data[2 * i]
                    x_odd = head_data[2 * i + 1]

                    new_head_data[2 * i] = x_even * c - x_odd * sin_val
                    new_head_data[2 * i + 1] = x_odd * c + x_even * sin_val
                seq_out.append(new_head_data)
            batch_out.append(seq_out)
        out.append(batch_out)
    return out
