from student.arch import compute_params


def compare_depth_vs_width(teacher_config, target_params):
    base_layers = teacher_config.get("num_hidden_layers", 12)
    base_hidden = teacher_config.get("hidden_size", 768)

    depth_config = dict(teacher_config)
    depth_config["num_hidden_layers"] = max(2, base_layers // 2)

    width_config = dict(teacher_config)
    new_hidden = int(base_hidden * 0.7071)
    new_hidden = (new_hidden // 64) * 64
    width_config["hidden_size"] = max(128, new_hidden)
    width_config["intermediate_size"] = 4 * width_config["hidden_size"]
    width_config["num_attention_heads"] = max(2, width_config["hidden_size"] // 64)
    width_config["num_key_value_heads"] = width_config["num_attention_heads"]

    return {
        "depth_student": depth_config,
        "width_student": width_config,
        "depth_params": compute_params(depth_config),
        "width_params": compute_params(width_config),
        "target_params": target_params,
    }
