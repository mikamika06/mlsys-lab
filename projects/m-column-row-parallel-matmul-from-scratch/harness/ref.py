import numpy as np


def get_test_cases():
    np.random.seed(123)
    cases = []
    for _ in range(5):
        b = np.random.randint(1, 4)
        s = np.random.randint(8, 32)
        in_dim = 16
        out_dim = 32
        x = np.random.randn(b, s, in_dim)
        w = np.random.randn(in_dim, out_dim)
        bias = np.random.randn(out_dim)
        cases.append((x, w, bias))
    return cases


def reference_column_parallel(x, w, bias):
    out = np.matmul(x, w)
    if bias is not None:
        out = out + bias
    return out


def reference_row_parallel(x, w, bias):
    out = np.matmul(x, w)
    if bias is not None:
        out = out + bias
    return out


def reference_communication_volume(batch_size, seq_len, hidden_dim, intermediate_dim, tp_size):
    tokens = batch_size * seq_len
    forward_bytes = 2 * tokens * intermediate_dim * 4
    backward_bytes = 2 * tokens * intermediate_dim * 4
    total_bytes = forward_bytes + backward_bytes
    return {
        "forward_bytes": forward_bytes,
        "backward_bytes": backward_bytes,
        "total_bytes": total_bytes
    }


def reference_dtensor_mlp(x, w1, w2, tp_size):
    hidden_dim = x.shape[-1]
    inter_dim = w1.shape[1]
    chunk_inter = inter_dim // tp_size

    outputs = []
    for i in range(tp_size):
        w1_chunk = w1[:, i * chunk_inter : (i + 1) * chunk_inter]
        h = np.matmul(x, w1_chunk)
        outputs.append(np.maximum(0, h))

    h_concatenated = np.concatenate(outputs, axis=-1)

    chunk_hidden = hidden_dim // tp_size
    mlp_outputs = []
    for i in range(tp_size):
        w2_chunk = w2[i * chunk_hidden : (i + 1) * chunk_hidden, :]
        out_chunk = np.matmul(h_concatenated[:, :, i * chunk_hidden : (i + 1) * chunk_hidden], w2_chunk)
        mlp_outputs.append(out_chunk)

    return sum(mlp_outputs)
