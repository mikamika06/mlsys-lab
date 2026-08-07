NVTX_ROWS = [
    {"name": "forward_pass", "start": 1000, "end": 5000},
    {"name": "attention_block", "start": 1500, "end": 3000},
    {"name": "mlp_block", "start": 3200, "end": 4800}
]

KERNEL_ROWS = [
    {"name": "gemm_fwd", "start": 1200, "end": 1800},
    {"name": "attn_fwd", "start": 1850, "end": 2600},
    {"name": "gemm_fwd", "start": 3300, "end": 3900},
    {"name": "gelu_fwd", "start": 4000, "end": 4300},
    {"name": "layer_norm", "start": 4400, "end": 4600}
]
