def softmax_memory_traffic(N, element_bytes, fused=False):
    elements = 2 * N if fused else 6 * N
    return elements * element_bytes

def achieved_bandwidth_GBps(N, element_bytes, time_ms, fused=False):
    traffic_bytes = softmax_memory_traffic(N, element_bytes, fused)
    time_s = time_ms / 1000.0
    return traffic_bytes / time_s / 1e9
