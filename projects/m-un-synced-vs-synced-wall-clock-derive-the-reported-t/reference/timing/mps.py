def measure_mps_synchronize_cost(synced_times, unsynced_times):
    """Measure real wall-clock cost of torch.mps.synchronize."""
    import statistics
    return statistics.mean(synced_times) - statistics.mean(unsynced_times)
