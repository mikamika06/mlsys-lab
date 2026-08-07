def generate_trace():
    return [
        {"ph": "X", "name": "aten::mm", "ts": 100, "dur": 50, "args": {"correlation_id": 1, "flops": 1000000}},
        {"ph": "X", "name": "cudaLaunchKernel", "ts": 110, "dur": 5, "args": {"correlation_id": 1}},
        {"ph": "X", "name": "cuLaunchKernel", "ts": 120, "dur": 2, "args": {}},
        {"ph": "X", "name": "gemm_kernel", "ts": 130, "dur": 40, "args": {"correlation_id": 1}},
        {"ph": "X", "name": "aten::relu", "ts": 200, "dur": 10, "args": {"correlation_id": 2}},
        {"ph": "X", "name": "elementwise_kernel", "ts": 215, "dur": 5, "args": {"correlation_id": 2}}
    ]

def pair_events(events):
    import ref
    from trace_parser.pairing import pair_events as pe
    return pe(events)

def classify_slices(events):
    from trace_parser.classify import classify_slices as cs
    return cs(events)

def compute_flops(events):
    from trace_parser.flops import compute_flops as cf
    return cf(events)
