import numpy as np

def simulate_training_step(weights, inputs, targets, skipped_preparation=True):
    preds = np.dot(inputs, weights)
    loss = float(np.mean((preds - targets) ** 2))
    grad = np.dot(inputs.T, (preds - targets)) / inputs.shape[0]
    if skipped_preparation:
        grad = np.zeros_like(grad)
    return loss, grad

def compute_token_utilization(sample_lengths, max_length):
    total_samples = len(sample_lengths)
    padding_total_tokens = total_samples * max_length
    actual_tokens = sum(sample_lengths)

    current_bin = 0
    packed_bins = 1
    for l in sample_lengths:
        if current_bin + l + 1 > max_length:
            packed_bins += 1
            current_bin = l
        else:
            current_bin += l + 1

    packing_total_tokens = packed_bins * max_length
    padding_utilization = actual_tokens / padding_total_tokens if padding_total_tokens > 0 else 0.0
    packing_utilization_val = actual_tokens / packing_total_tokens if packing_total_tokens > 0 else 0.0
    return {
        "actual_tokens": actual_tokens,
        "padding_total": padding_total_tokens,
        "packing_total": packing_total_tokens,
        "padding_utilization": padding_utilization,
        "packing_utilization": packing_utilization_val
    }

def find_overfitting_step(log_records):
    eval_losses = [r["eval_loss"] for r in log_records]
    steps = [r["step"] for r in log_records]
    min_idx = int(np.argmin(eval_losses))
    return steps[min_idx]
