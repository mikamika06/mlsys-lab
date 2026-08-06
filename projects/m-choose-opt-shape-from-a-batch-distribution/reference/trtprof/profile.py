import numpy as np


def select_opt_shape(batch_samples, strategy="p50"):
    """Select min, opt, max dynamic shape bounds from batch observations."""
    samples = np.asarray(batch_samples, dtype=np.int64)
    if samples.size == 0:
        raise ValueError("batch_samples cannot be empty")
    
    if strategy == "p50":
        opt_val = int(np.percentile(samples, 50))
    elif strategy == "p90":
        opt_val = int(np.percentile(samples, 90))
    elif strategy == "mode":
        vals, counts = np.unique(samples, return_counts=True)
        opt_val = int(vals[np.argmax(counts)])
    elif strategy == "mean":
        opt_val = int(np.round(np.mean(samples)))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return opt_val


def calculate_profile_bounds(batch_samples, strategy="p50", padding_ratio=0.1):
    """Calculate (min_shape, opt_shape, max_shape) for a tensor dimension."""
    samples = np.asarray(batch_samples, dtype=np.int64)
    opt_val = select_opt_shape(samples, strategy=strategy)
    
    min_val = int(np.min(samples))
    max_val = int(np.max(samples))
    
    if padding_ratio > 0:
        pad = int(np.ceil((max_val - min_val) * padding_ratio))
        min_val = max(1, min_val - pad)
        max_val = max_val + pad
    
    min_val = min(min_val, opt_val)
    max_val = max(max_val, opt_val)
    
    return (min_val, opt_val, max_val)
