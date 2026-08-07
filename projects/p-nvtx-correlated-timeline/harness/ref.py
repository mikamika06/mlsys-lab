def get_sample_trace():
    return [
        {"name": "preprocessing", "start": 0, "end": 15},
        {"name": "inference", "start": 20, "end": 80},
        {"name": "postprocessing", "start": 85, "end": 100}
    ]

def get_sample_kernels():
    return [
        {"start": 25, "end": 40, "name": "gemm_kernel"},
        {"start": 45, "end": 70, "name": "attn_kernel"}
    ]

def get_second_trace():
    return [
        {"name": "preprocessing", "start": 0, "end": 10},
        {"name": "inference", "start": 12, "end": 90},
        {"name": "postprocessing", "start": 92, "end": 105}
    ]
