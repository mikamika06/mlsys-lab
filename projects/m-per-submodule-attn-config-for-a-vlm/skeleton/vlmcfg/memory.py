def plan_bytes(config, seq_len, dtype_size, batch_size):
    raise NotImplementedError()

def uniform_bytes(config, seq_len, dtype_size, batch_size):
    raise NotImplementedError()

def free_schedule(seq_len, dtype_size, step_count):
    raise NotImplementedError()
