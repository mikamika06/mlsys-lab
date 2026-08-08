def column_parallel_matmul(x, weight, bias=None):
    raise NotImplementedError


def row_parallel_matmul(x, weight, bias=None):
    raise NotImplementedError


def tp_communication_volume(batch_size, seq_len, hidden_dim, intermediate_dim, tp_size):
    raise NotImplementedError


def dtensor_mlp(x, w1, w2, tp_size):
    raise NotImplementedError
