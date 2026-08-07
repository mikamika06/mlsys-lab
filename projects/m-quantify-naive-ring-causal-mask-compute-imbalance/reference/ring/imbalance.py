def analyze_naive_ring(num_devices: int):
    out = []
    for i in range(num_devices):
        out.append({
            "rank": i,
            "fully_unmasked": i,
            "partially_unmasked": 1,
            "fully_masked": num_devices - 1 - i
        })
    return out
