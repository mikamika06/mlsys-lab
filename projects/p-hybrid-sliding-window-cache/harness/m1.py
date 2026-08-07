import sys


def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from kvcache.config import LayerConfig, ModelConfig

    m = {"config_parsed": 0.0, "layer_types_valid": 0.0, "window_sizes_valid": 0.0}

    try:
        raw_dict = {
            "num_layers": 4,
            "num_heads": 8,
            "head_dim": 64,
            "layer_configs": [
                {"layer_id": 0, "is_sliding": False},
                {"layer_id": 1, "is_sliding": True, "window_size": 128},
                {"layer_id": 2, "is_sliding": False},
                {"layer_id": 3, "is_sliding": True, "window_size": 256},
            ],
        }
        cfg = ModelConfig.from_dict(raw_dict)
        if (
            cfg.num_layers == 4
            and cfg.num_heads == 8
            and cfg.head_dim == 64
            and len(cfg.layer_configs) == 4
        ):
            m["config_parsed"] = 1.0
    except Exception:
        return m

    if (
        not cfg.layer_configs[0].is_sliding
        and cfg.layer_configs[1].is_sliding
        and not cfg.layer_configs[2].is_sliding
        and cfg.layer_configs[3].is_sliding
    ):
        m["layer_types_valid"] = 1.0

    if (
        cfg.layer_configs[1].window_size == 128
        and cfg.layer_configs[3].window_size == 256
        and cfg.layer_configs[0].window_size is None
    ):
        m["window_sizes_valid"] = 1.0

    try:
        LayerConfig(layer_id=0, is_sliding=True, window_size=None)
        m["window_sizes_valid"] = 0.0
    except ValueError:
        pass

    return m
