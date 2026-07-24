def _oracle(num_devices, seq_per_device, hidden_dim, bytes_per_element):
    kv_bytes = 2 * seq_per_device * hidden_dim * bytes_per_element
    total = num_devices * (num_devices - 1) * kv_bytes
    return kv_bytes, total


def grade(sol, fx) -> dict:
    cases = [
        (2, 1, 8, 2),
        (4, 1024, 4096, 2),
        (8, 512, 8192, 2),
        (16, 128, 4096, 4),
        (3, 77, 1536, 2),
    ]

    ok = 1.0
    for args in cases:
        try:
            got = sol.ring_attention_comm(*args)
        except Exception:
            ok = 0.0
            break

        if tuple(got) != _oracle(*args):
            ok = 0.0
            break

    return {"modeled_mem_access": ok}
