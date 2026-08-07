import ref

def configure_arena(strategy, initial_chunk):
    cfg = ref.oracle_arena_config(strategy)
    cfg["chunk_size"] = initial_chunk
    return cfg
