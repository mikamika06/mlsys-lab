def generate_traces():
    trace_a = [
        {"name": "gemm_kernel", "duration": 10.0, "type": "kernel"},
        {"name": "elemwise_add", "duration": 2.0, "type": "kernel"},
    ]
    trace_b = [
        {"name": "gemm_kernel", "duration": 10.0, "type": "kernel"},
        {"name": "elemwise_add", "duration": 2.0, "type": "kernel"},
        {"name": "sync_barrier", "duration": 50.0, "type": "sync", "gap": 150},
    ]
    trace_c = [
        {"name": "gemm_kernel", "duration": 10.0, "type": "kernel"},
        {"name": "elemwise_add", "duration": 2.0, "type": "kernel"},
    ]
    return trace_a, trace_b, trace_c
