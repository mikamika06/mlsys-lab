import numpy as np

CONFIGS = [
    {
        "model_cfg": {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "overhead_ratio": 0.15},
        "prefixes": [
            {"id": "p1", "token_count": 4096, "last_accessed_hours_ago": 2.5, "sharing_count": 3},
            {"id": "p2", "token_count": 8192, "last_accessed_hours_ago": 26.0, "sharing_count": 1},
            {"id": "p3", "token_count": 1024, "last_accessed_hours_ago": 12.0, "sharing_count": 2},
        ],
        "retention_hours": 24.0,
    },
    {
        "model_cfg": {"num_layers": 80, "num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "overhead_ratio": 0.08},
        "prefixes": [
            {"id": "pA", "token_count": 16384, "last_accessed_hours_ago": 1.0, "sharing_count": 5},
            {"id": "pB", "token_count": 32768, "last_accessed_hours_ago": 18.0, "sharing_count": 1},
        ],
        "retention_hours": 24.0,
    },
    {
        "model_cfg": {"num_layers": 40, "num_kv_heads": 16, "head_dim": 128, "dtype_bytes": 2, "overhead_ratio": 0.10},
        "prefixes": [
            {"id": "pX", "token_count": 2048, "last_accessed_hours_ago": 30.0, "sharing_count": 10},
        ],
        "retention_hours": 24.0,
    }
]

DISK_WORKLOADS = [
    {
        "block_size": 4096,
        "requests": [
            {"offset_bytes": 0, "length_bytes": 4096},
            {"offset_bytes": 100, "length_bytes": 500},
            {"offset_bytes": 8100, "length_bytes": 1000},
        ]
    },
    {
        "block_size": 65536,
        "requests": [
            {"offset_bytes": 10, "length_bytes": 60000},
            {"offset_bytes": 120000, "length_bytes": 10000},
        ]
    },
    {
        "block_size": 512,
        "requests": [
            {"offset_bytes": 512, "length_bytes": 512},
            {"offset_bytes": 1024, "length_bytes": 1024},
        ]
    }
]

def build_plan(model_cfg, prefixes, retention_hours):
    layers = model_cfg["num_layers"]
    heads = model_cfg["num_kv_heads"]
    hdim = model_cfg["head_dim"]
    dtype_b = model_cfg.get("dtype_bytes", 2)
    bytes_per_token = 2 * layers * heads * hdim * dtype_b
    
    total_tokens = 0
    retained_prefixes = 0
    for p in prefixes:
        if p["last_accessed_hours_ago"] <= retention_hours:
            total_tokens += p["token_count"] * p.get("sharing_count", 1)
            retained_prefixes += 1
            
    base_bytes = total_tokens * bytes_per_token
    overhead_pct = model_cfg.get("overhead_ratio", 0.10)
    total_bytes = int(base_bytes * (1.0 + overhead_pct))
    
    return {
        "retained_prefixes": retained_prefixes,
        "total_tokens": total_tokens,
        "base_kv_bytes": base_bytes,
        "total_ram_bytes": total_bytes
    }

def build_disk_measure(block_size_bytes, requests):
    logical_bytes = 0
    physical_bytes = 0
    
    for req in requests:
        offset = req["offset_bytes"]
        length = req["length_bytes"]
        logical_bytes += length
        
        start_block = offset // block_size_bytes
        end_block = (offset + length - 1) // block_size_bytes
        num_blocks = (end_block - start_block) + 1
        physical_bytes += num_blocks * block_size_bytes
        
    amp_factor = (physical_bytes / logical_bytes) if logical_bytes > 0 else 1.0
    return {
        "logical_bytes": logical_bytes,
        "physical_bytes": physical_bytes,
        "read_amplification": amp_factor
    }
