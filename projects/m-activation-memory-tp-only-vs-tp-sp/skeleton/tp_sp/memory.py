def activation_memory_per_layer(seq_len: int, batch_size: int, hidden_size: int, tp_size: int, use_sp: bool) -> int:
    raise NotImplementedError

def forward_communication_volume(seq_len: int, batch_size: int, hidden_size: int, tp_size: int, use_sp: bool) -> int:
    raise NotImplementedError
