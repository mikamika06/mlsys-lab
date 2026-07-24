def ring_attention_comm(num_devices, seq_per_device, hidden_dim, bytes_per_element):
    kv_bytes = 2 * seq_per_device * hidden_dim * bytes_per_element
    total_bytes = num_devices * (num_devices - 1) * kv_bytes
    return kv_bytes, total_bytes
