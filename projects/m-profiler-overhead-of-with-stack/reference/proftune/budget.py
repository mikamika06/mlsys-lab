def select_config_for_budget(step_meta, byte_budget):
    if byte_budget < 5000:
        return {"with_stack": False, "record_shapes": False, "profile_memory": False}
    elif byte_budget < 20000:
        return {"with_stack": True, "record_shapes": False, "profile_memory": False}
    else:
        return {"with_stack": True, "record_shapes": True, "profile_memory": True}
