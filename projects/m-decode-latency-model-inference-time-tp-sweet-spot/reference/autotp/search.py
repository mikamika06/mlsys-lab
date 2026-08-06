from autotp.model import estimate_decode_latency

def find_tp_sweet_spot(config, hw, batch_size, max_tp=8):
    best_tp = 1
    best_lat = float("inf")
    for tp in [1, 2, 4, 8]:
        if tp > max_tp:
            continue
        lat = estimate_decode_latency(config, hw, tp, batch_size)
        if lat < best_lat:
            best_lat = lat
            best_tp = tp
    return best_tp
