def measure_tokens_per_sec(batch_sizes, latencies, token_counts):
    results = {}
    for b, lat, tokens in zip(batch_sizes, latencies, token_counts):
        if lat <= 0:
            results[b] = 0.0
        else:
            results[b] = float(tokens) / float(lat)
    return results
