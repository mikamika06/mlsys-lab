import numpy as np


def column_parallel_matmul(x, weight, bias=None):
    out = np.matmul(x, weight)
    if bias is not None:
        out = out + bias
    return out


def row_parallel_matmul(x, weight, bias=None):
    out = np.matmul(x, weight)
    if bias is not None:
        out = out + bias
    return out


def tp_communication_volume(batch_size, seq_len, hidden_dim, intermediate_dim, tp_size):
    tokens = batch_size * seq_len
    forward_bytes = 2 * tokens * intermediate_dim * 4
    backward_bytes = 2 * tokens * intermediate_dim * 4
    total_bytes = forward_bytes + backward_bytes
    return {
        "forward_bytes": forward_bytes,
        "backward_bytes": backward_bytes,
        "total_bytes": total_bytes
    }


def dtensor_mlp(x, w1, w2, tp_size):
    hidden_dim = x.shape[-1]
    inter_dim = w1.shape[1] * tp_size
    chunk_inter = inter_dim // tp_size

    outputs = []
    for i in range(tp_size):
        w1_chunk = w1[:, i * chunk_inter : (i + 1) * chunk_inter]
        h = column_parallel_matmul(x, w1_chunk)
        h_activated = np.maximum(0, h)
        outputs.append(h_activated)

    h_concatenated = np.concatenate(outputs, axis=-1)

    chunk_hidden = hidden_dim // tp_size
    mlp_outputs = []
    for i in range(tp_size):
        w2_chunk = w2[i * chunk_hidden : (i + 1) * chunk_hidden, :]
        out_chunk = row_parallel_matmul(h_concatenated[:, :, i * chunk_hidden : (i + 1) * chunk_hidden], w2_chunk)
        mlp_outputs.append(out_chunk)

    final_output = sum(mlp_outputs)
    return final_output
