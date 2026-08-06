def ring_attention_comm(num_devices, seq_per_device, hidden_dim, bytes_per_element):
    kv_bytes = 2 * seq_per_device * hidden_dim * bytes_per_element

    total_dense = num_devices * (num_devices - 1) * kv_bytes

    per_device = []
    for i in range(num_devices):
        forwarded = i + 1 if i < num_devices - 1 else 0
        per_device.append(forwarded * kv_bytes)

    total_causal = sum(per_device)

    return kv_bytes, total_dense, total_causal, tuple(per_device)
