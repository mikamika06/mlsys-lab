def compute_buffer_bytes(batch, ubatch, hidden, ffn):
    return ubatch * (hidden + ffn) * 4

def load_time_seconds(model_bytes, use_mmap, disk_bw):
    if use_mmap:
        return 0.05
    return model_bytes / disk_bw
