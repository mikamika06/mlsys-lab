def activation_memory_per_layer(seq_len: int, batch_size: int, hidden_size: int, tp_size: int, use_sp: bool) -> int:
    base = seq_len * batch_size * hidden_size
    if use_sp:
        return base // tp_size
    return base

def forward_communication_volume(seq_len: int, batch_size: int, hidden_size: int, tp_size: int, use_sp: bool) -> int:
    v = seq_len * batch_size * hidden_size
    return 2 * v * (tp_size - 1) // tp_size
