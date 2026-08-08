from .pruner import prune_model


def _orig_flops_params(config):
    total_params = 0
    total_flops = 0
    feat_size = config.get("input_feature_size", 32)
    for l in config["layers"]:
        ltype = l["type"]
        in_c = l.get("in_channels", 0)
        out_c = l.get("out_channels", 0)
        k = l.get("kernel_size", 1)
        if ltype == "conv":
            p = in_c * out_c * k * k + (out_c if l.get("bias", False) else 0)
            f = 2 * in_c * out_c * k * k * feat_size * feat_size
        elif ltype == "bn":
            p = 2 * out_c
            f = 2 * out_c * feat_size * feat_size
        elif ltype == "linear":
            p = in_c * out_c + (out_c if l.get("bias", True) else 0)
            f = 2 * in_c * out_c
        else:
            p, f = 0, 0
        total_params += p
        total_flops += f
    return total_params, total_flops


def simulate_speedup_gap(config, pruning_plan, unstructured_sparsity):
    orig_p, orig_f = _orig_flops_params(config)
    pruned_res = prune_model(config, pruning_plan)
    struct_f = pruned_res["total_flops"]

    struct_flop_ratio = orig_f / max(1, struct_f)
    struct_measured_speedup = struct_flop_ratio * 0.92

    unstruct_flop_ratio = 1.0 / max(0.01, 1.0 - unstructured_sparsity)
    sparse_overhead_factor = 1.4 + (unstructured_sparsity * 0.8)
    unstruct_measured_speedup = unstruct_flop_ratio / sparse_overhead_factor

    speedup_gap = struct_measured_speedup / max(0.01, unstruct_measured_speedup)

    return {
        "orig_flops": orig_f,
        "structured_flops": struct_f,
        "structured_speedup": struct_measured_speedup,
        "unstructured_speedup": unstruct_measured_speedup,
        "speedup_gap": speedup_gap,
    }
