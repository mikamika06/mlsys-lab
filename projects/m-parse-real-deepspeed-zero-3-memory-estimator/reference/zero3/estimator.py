import math


def estimate_zero3_memory(layer_specs, world_size, alignment_bytes=512, bytes_per_elem=2):
    elems_per_align = alignment_bytes // bytes_per_elem
    total_sharded_params = 0
    max_unsharded_layer = 0

    for layer in layer_specs:
        layer_params = 0
        for shape in layer["params"]:
            numel = 1
            for d in shape:
                numel *= d
            layer_params += numel
        padded_layer_params = math.ceil(layer_params / (world_size * elems_per_align)) * (world_size * elems_per_align)
        sharded_per_rank = padded_layer_params // world_size
        total_sharded_params += sharded_per_rank
        if layer_params > max_unsharded_layer:
            max_unsharded_layer = layer_params

    sharded_bytes = total_sharded_params * bytes_per_elem
    unsharded_peak_bytes = max_unsharded_layer * bytes_per_elem
    fp32_optimizer_bytes = total_sharded_params * 4 * 3
    grad_bytes = total_sharded_params * bytes_per_elem

    return {
        "sharded_param_bytes": sharded_bytes,
        "unsharded_param_peak_bytes": unsharded_peak_bytes,
        "optimizer_state_bytes": fp32_optimizer_bytes,
        "gradient_bytes": grad_bytes,
        "total_static_bytes": sharded_bytes + unsharded_peak_bytes + fp32_optimizer_bytes + grad_bytes,
    }


def calculate_peak_forward_memory(layer_specs, world_size, prefetch_depth=1, bytes_per_elem=2):
    num_layers = len(layer_specs)
    layer_unsharded_bytes = []
    layer_sharded_bytes = []

    for layer in layer_specs:
        tot = sum(math.prod(s) for s in layer["params"])
        padded = math.ceil(tot / world_size) * world_size
        layer_unsharded_bytes.append(tot * bytes_per_elem)
        layer_sharded_bytes.append((padded // world_size) * bytes_per_elem)

    base_sharded_bytes = sum(layer_sharded_bytes)
    peak_active = 0

    for i in range(num_layers):
        activations = sum(layer_specs[j].get("activation_bytes", 0) for j in range(i + 1))
        active_range_end = min(num_layers, i + 1 + prefetch_depth)
        gathered_bytes = sum(layer_unsharded_bytes[j] for j in range(i, active_range_end))
        current_peak = base_sharded_bytes + gathered_bytes + activations
        if current_peak > peak_active:
            peak_active = current_peak

    return peak_active
