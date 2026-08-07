import math

def masked_softmax(logits: list[list[float]], mask: list[list[bool]]) -> list[list[float]]:
    result = []
    for row_logits, row_mask in zip(logits, mask):
        masked_row = []
        for val, m in zip(row_logits, row_mask):
            if m:
                masked_row.append(float('-inf'))
            else:
                masked_row.append(float(val))

        max_val = -float('inf')
        for val in masked_row:
            if val > max_val:
                max_val = val

        exp_row = []
        sum_exp = 0.0
        for val in masked_row:
            if val == float('-inf'):
                exp_row.append(0.0)
            else:
                e = math.exp(val - max_val)
                exp_row.append(e)
                sum_exp += e

        if sum_exp == 0.0:
            norm_row = [0.0 for _ in masked_row]
        else:
            norm_row = [e / sum_exp for e in exp_row]

        result.append(norm_row)
    return result
