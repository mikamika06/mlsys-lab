def compute_buffer_bytes(batch, ubatch, hidden, ffn):
    return ubatch * (hidden + ffn) * 4

def load_time_seconds(model_bytes, use_mmap, disk_bw):
    if use_mmap:
        return 0.05
    return model_bytes / disk_bw

def attribute_oom(ram_total, model_bytes, kv_bytes, compute_bytes, use_mmap):
    avail = ram_total
    if not use_mmap:
        if avail < model_bytes:
            return "weights"
        avail -= model_bytes
    if avail < kv_bytes:
        return "kv"
    avail -= kv_bytes
    if avail < compute_bytes:
        return "compute"
    return "none"

FIXTURES_PREDICT = [
    (1024, 512, 4096, 11008),
    (2048, 2048, 4096, 11008),
    (512, 128, 2048, 5504)
]

FIXTURES_LOAD = [
    (7000000000, True, 100000000),
    (7000000000, False, 100000000),
    (13000000000, True, 50000000),
    (13000000000, False, 50000000),
]

FIXTURES_OOM = [
    (10, 5, 2, 1, False),
    (10, 11, 2, 1, False),
    (10, 11, 2, 1, True),
    (10, 5, 6, 1, False),
    (10, 5, 6, 1, True),
    (10, 0, 11, 1, True),
    (10, 5, 2, 4, False),
    (10, 5, 2, 9, True),
]
