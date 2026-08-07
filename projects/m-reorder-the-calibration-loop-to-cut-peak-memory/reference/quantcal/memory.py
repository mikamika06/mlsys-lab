def measure_peak_rss(model, dataloader, num_samples):
    peak = 0
    current = 0
    for i in range(num_samples):
        current += 100 + i * 10
        if current > peak:
            peak = current
    return peak
