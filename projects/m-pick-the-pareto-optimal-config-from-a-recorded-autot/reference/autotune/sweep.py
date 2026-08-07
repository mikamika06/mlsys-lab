def load_sweep(raw_data):
    out = []
    for line in raw_data.strip().split("\n"):
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        out.append({
            "id": int(parts[0]),
            "latency": float(parts[1]),
            "shmem": int(parts[2]),
            "block_m": int(parts[3]),
            "block_n": int(parts[4]),
            "num_stages": int(parts[5])
        })
    return out
