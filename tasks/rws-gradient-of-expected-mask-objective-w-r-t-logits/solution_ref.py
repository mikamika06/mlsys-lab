import math


def expected_mask_grad(logits, values, target):
    batch_size = len(logits)
    num_logits = len(logits[0])
    num_values = len(values)

    out = [[0.0 for _ in range(num_logits)] for _ in range(batch_size)]

    for i in range(batch_size):
        max_val = logits[i][0]
        for j in range(1, num_logits):
            if logits[i][j] > max_val:
                max_val = logits[i][j]

        exp_logits_row = [0.0 for _ in range(num_logits)]
        sum_exp = 0.0
        for j in range(num_logits):
            val = math.exp(logits[i][j] - max_val)
            exp_logits_row[j] = val
            sum_exp += val

        probs_row = [0.0 for _ in range(num_logits)]
        for j in range(num_logits):
            probs_row[j] = exp_logits_row[j] / sum_exp

        mask_i = 0.0
        for j in range(num_values):
            mask_i += probs_row[j] * values[j]

        d_loss_d_mask_i = 2.0 * (mask_i - target[i])

        for j in range(num_logits):
            centered_val = values[j] - mask_i
            out[i][j] = d_loss_d_mask_i * probs_row[j] * centered_val

    return out
