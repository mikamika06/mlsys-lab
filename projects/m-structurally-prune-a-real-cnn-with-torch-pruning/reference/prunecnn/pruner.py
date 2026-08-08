from .depgraph import get_pruning_group


def _calc_layer_params_flops(layer, feat_size):
    ltype = layer["type"]
    in_c = layer.get("in_channels", 0)
    out_c = layer.get("out_channels", 0)
    k = layer.get("kernel_size", 1)

    if ltype == "conv":
        params = in_c * out_c * k * k + (out_c if layer.get("bias", False) else 0)
        flops = 2 * in_c * out_c * k * k * feat_size * feat_size
    elif ltype == "bn":
        params = 2 * out_c
        flops = 2 * out_c * feat_size * feat_size
    elif ltype == "linear":
        params = in_c * out_c + (out_c if layer.get("bias", True) else 0)
        flops = 2 * in_c * out_c
    elif ltype == "add":
        params = 0
        flops = in_c * feat_size * feat_size
    else:
        params = 0
        flops = 0
    return params, flops


def prune_model(config, pruning_plan):
    layers = [dict(l) for l in config["layers"]]
    layer_map = {l["name"]: l for l in layers}

    combined_group = {}
    for trigger_layer, channels in pruning_plan.items():
        grp = get_pruning_group(config, trigger_layer, channels)
        for lname, dirs in grp.items():
            if lname not in combined_group:
                combined_group[lname] = {"in": set(), "out": set()}
            combined_group[lname]["in"].update(dirs["in"])
            combined_group[lname]["out"].update(dirs["out"])

    for lname, dirs in combined_group.items():
        layer = layer_map[lname]
        prune_in = dirs["in"]
        prune_out = dirs["out"]

        if layer["type"] in ("conv", "linear"):
            if prune_in:
                layer["in_channels"] = max(1, layer["in_channels"] - len(prune_in))
            if prune_out:
                layer["out_channels"] = max(1, layer["out_channels"] - len(prune_out))
        elif layer["type"] == "bn":
            if prune_out or prune_in:
                ch_pruned = len(prune_out or prune_in)
                layer["out_channels"] = max(1, layer["out_channels"] - ch_pruned)
                layer["in_channels"] = layer["out_channels"]
        elif layer["type"] == "add":
            if prune_out or prune_in:
                ch_pruned = len(prune_out or prune_in)
                layer["out_channels"] = max(1, layer["out_channels"] - ch_pruned)
                layer["in_channels"] = layer["out_channels"]

    total_params = 0
    total_flops = 0
    feat_size = config.get("input_feature_size", 32)

    for l in layers:
        p, f = _calc_layer_params_flops(l, feat_size)
        total_params += p
        total_flops += f

    return {
        "layers": layers,
        "total_params": total_params,
        "total_flops": total_flops,
    }
