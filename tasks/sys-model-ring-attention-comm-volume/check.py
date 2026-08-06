def _oracle(num_devices, seq_per_device, hidden_dim, bytes_per_element):
    kv_bytes = 2 * seq_per_device * hidden_dim * bytes_per_element

    dense = num_devices * (num_devices - 1) * kv_bytes

    per_device = []
    for i in range(num_devices):
        blocks_forwarded = i + 1 if i < num_devices - 1 else 0
        per_device.append(blocks_forwarded * kv_bytes)

    causal = sum(per_device)

    return kv_bytes, dense, causal, tuple(per_device)


def grade(sol, fx) -> dict:
    cases = [
        (2, 1, 8, 2),
        (4, 1024, 4096, 2),
        (8, 512, 8192, 2),
        (16, 128, 4096, 4),
        (3, 77, 1536, 2),
        (7, 1000, 768, 1),
    ]

    for args in cases:
        try:
            got = sol.ring_attention_comm(*args)
        except Exception:
            return {"modeled_mem_access": 0.0}

        try:
            got = tuple(got)
        except TypeError:
            return {"modeled_mem_access": 0.0}

        if len(got) != 4:
            return {"modeled_mem_access": 0.0}

        try:
            head = tuple(int(v) for v in got[:3])
            tail = tuple(int(v) for v in got[3])
        except (TypeError, ValueError):
            return {"modeled_mem_access": 0.0}

        if any(isinstance(v, bool) or int(v) != v for v in got[:3]):
            return {"modeled_mem_access": 0.0}
        if len(tail) != args[0]:
            return {"modeled_mem_access": 0.0}

        ref = _oracle(*args)
        if head != ref[:3] or tail != ref[3]:
            return {"modeled_mem_access": 0.0}

    return {"modeled_mem_access": 1.0}
