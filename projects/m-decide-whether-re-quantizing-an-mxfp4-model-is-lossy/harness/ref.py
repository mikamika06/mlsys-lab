CONFIGS = [
    {"source_format": "mxfp4", "target_format": "mxfp4", "block_size": 32, "target_block_size": 32},
    {"source_format": "mxfp4", "target_format": "mxfp4", "block_size": 32, "target_block_size": 64},
    {"source_format": "fp16", "target_format": "mxfp4", "block_size": 16, "target_block_size": 32},
    {"source_format": "mxfp4", "target_format": "mxfp4", "block_size": 64, "target_block_size": 64},
    {"source_format": "bf16", "target_format": "mxfp4", "block_size": 32, "target_block_size": 32},
]

SPECS = [
    {
        "layers": [
            {
                "type": "moe",
                "num_experts": 4,
                "expert_params": 128,
                "router_params": 64,
                "block_size": 32
            }
        ]
    },
    {
        "layers": [
            {
                "type": "dense",
                "params": 256
            },
            {
                "type": "moe",
                "num_experts": 2,
                "expert_params": 256,
                "router_params": 128,
                "block_size": 32
            }
        ]
    }
]
