from kvpage.unify import find_common_block_size

CONFIG = {
    "layers": [
        {"index": 0, "kv_heads": 1, "head_dim": 3, "dtype_bytes": 1},
        {"index": 1, "kv_heads": 8, "head_dim": 64, "dtype_bytes": 2},
    ]
}


def test_common_block_size_alignment():
    candidates = (8, 16, 32, 64)
    align = 64
    chosen = find_common_block_size(CONFIG, candidates, align)
    for layer in CONFIG["layers"]:
        bytes_per_token = 2 * layer["kv_heads"] * layer["head_dim"] * layer.get("dtype_bytes", 2)
        block_bytes = chosen * bytes_per_token
        assert block_bytes % align == 0, f"Layer {layer['index']} block size {block_bytes} not aligned to {align}"
