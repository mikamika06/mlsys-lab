import math

def sdpa(Q: list[list[float]], K: list[list[float]], V: list[list[float]]) -> list[list[float]]:
    batch = len(Q)
    d_k = len(Q[0])
    d_v = len(V[0])
    scale = math.sqrt(d_k)
    out_list = []
    for i in range(batch):
        row_logits = []
        for j in range(d_k):
            dot_val = 0.0
            for k in range(d_k):
                dot_val += float(Q[i][k]) * float(K[j][k])
            row_logits.append(dot_val / scale)
        max_val = row_logits[0]
        for val in row_logits:
            if val > max_val:
                max_val = val
        row_exp = []
        sum_exp = 0.0
        for val in row_logits:
            e = math.exp(val - max_val)
            row_exp.append(e)
            sum_exp += e
        row_softmax = [e / sum_exp for e in row_exp]
        row_out = []
        for j in range(d_v):
            out_val = 0.0
            for k in range(d_k):
                out_val += row_softmax[k] * float(V[k][j])
            row_out.append(out_val)
        out_list.append(row_out)
    return out_list
