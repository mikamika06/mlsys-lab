CONFIGS = [
    (1024, 2, 512, 4),
    (2048, 4, 1024, 8),
    (4096, 1, 4096, 2),
    (8192, 8, 2048, 8)
]

def activation_memory_per_layer(seq_len, batch_size, hidden_size, tp_size, use_sp):
    base = seq_len * batch_size * hidden_size
    if use_sp:
        return base // tp_size
    return base

def forward_communication_volume(seq_len, batch_size, hidden_size, tp_size, use_sp):
    v = seq_len * batch_size * hidden_size
    return 2 * v * (tp_size - 1) // tp_size
