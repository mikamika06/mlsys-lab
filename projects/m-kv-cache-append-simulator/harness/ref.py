import numpy as np

def append_tokens(cache_seqlens, block_tables, block_size):
    res = []
    for seqlen, table in zip(cache_seqlens, block_tables):
        block_idx = seqlen // block_size
        block_offset = seqlen % block_size
        res.append((table[block_idx], block_offset))
    return res

def decode_bandwidth(cache_seqlens, num_layers, num_kv_heads, head_dim, dtype_bytes):
    read_tokens = sum(cache_seqlens)
    write_tokens = len(cache_seqlens)
    bytes_per_token = 2 * num_kv_heads * head_dim * dtype_bytes
    layer_bytes = (read_tokens + write_tokens) * bytes_per_token
    return layer_bytes * num_layers

def generate_fixtures():
    np.random.seed(42)
    fixtures = []
    for _ in range(50):
        bsz = np.random.randint(1, 32)
        block_size = int(np.random.choice([16, 32, 64]))
        cache_seqlens = np.random.randint(1, 1024, size=bsz).tolist()
        block_tables = []
        for seqlen in cache_seqlens:
            num_blocks_needed = (seqlen // block_size) + 1
            table = np.random.randint(0, 10000, size=num_blocks_needed).tolist()
            block_tables.append(table)
        num_layers = np.random.randint(1, 32)
        num_kv_heads = int(np.random.choice([8, 32]))
        head_dim = int(np.random.choice([64, 128]))
        dtype_bytes = 2
        fixtures.append({
            "cache_seqlens": cache_seqlens,
            "block_tables": block_tables,
            "block_size": block_size,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype_bytes": dtype_bytes
        })
    return fixtures

FIXTURES = generate_fixtures()
