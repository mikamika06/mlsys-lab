def compute_theoretical_volume(model_config, world_size):
    raise NotImplementedError

def find_barriers(events):
    raise NotImplementedError

def optimize_buckets(layers, target_bucket_size):
    raise NotImplementedError

def measure_unoverlapped_ratio(timeline):
    raise NotImplementedError

def predict_scaling(base_time, comm_volume, world_size):
    raise NotImplementedError
