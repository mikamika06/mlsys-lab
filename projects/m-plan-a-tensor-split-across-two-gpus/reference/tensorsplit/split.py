from tensorsplit.sizes import compute_layer_sizes


def compute_tensor_split(config):
    sizes = compute_layer_sizes(config)
    total = sum(sizes)
    if total == 0:
        return [0.5, 0.5]
    target = total / 2.0
    current = 0.0
    split_idx = len(sizes)
    for i, s in enumerate(sizes):
        if current + s > target and abs((current + s) - target) >= abs(current - target):
            split_idx = i
            break
        current += s
    sum0 = sum(sizes[:split_idx])
    sum1 = sum(sizes[split_idx:])
    if sum0 == 0 or sum1 == 0:
        split_idx = max(1, len(sizes) // 2)
        sum0 = sum(sizes[:split_idx])
        sum1 = sum(sizes[split_idx:])
    return [sum0 / total, sum1 / total]
