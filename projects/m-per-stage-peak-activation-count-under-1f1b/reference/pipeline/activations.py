def compute_peak_activations(num_stages, num_microbatches, stage):
    fwd_in_flight = min(num_microbatches, num_stages - stage + 2)
    return max(1, fwd_in_flight)
