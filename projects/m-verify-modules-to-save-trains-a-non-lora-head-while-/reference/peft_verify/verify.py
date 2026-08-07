def check_modules_to_save_trainable(model, modules_to_save):
    base_frozen = True
    head_trainable = True

    for name, param in model.named_parameters():
        is_saved_module = any(m_name in name for m_name in modules_to_save)
        is_lora_param = "lora_" in name

        if is_saved_module:
            if not param.requires_grad:
                head_trainable = False
        elif is_lora_param:
            if not param.requires_grad:
                base_frozen = False
        else:
            if param.requires_grad:
                base_frozen = False

    return {
        "base_frozen": base_frozen,
        "head_trainable": head_trainable,
        "valid": base_frozen and head_trainable,
    }
