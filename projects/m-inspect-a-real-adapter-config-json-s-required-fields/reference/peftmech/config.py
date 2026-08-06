class AdapterConfigError(Exception):
    pass


REQUIRED_FIELDS = {
    "peft_type": str,
    "r": int,
    "lora_alpha": (int, float),
    "target_modules": (list, set),
    "bias": str,
}

DEFAULTS = {
    "peft_type": "LORA",
    "r": 8,
    "lora_alpha": 16.0,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
}


def validate_adapter_config(config_dict):
    if not isinstance(config_dict, dict):
        raise AdapterConfigError("Config must be a dictionary")
    missing = []
    invalid_types = []
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in config_dict or config_dict[field] is None:
            missing.append(field)
        elif not isinstance(config_dict[field], expected_type):
            invalid_types.append(field)
    if missing or invalid_types:
        errs = []
        if missing:
            errs.append(f"Missing required fields: {sorted(missing)}")
        if invalid_types:
            errs.append(f"Invalid field types: {sorted(invalid_types)}")
        raise AdapterConfigError("; ".join(errs))
    if config_dict["peft_type"].upper() != "LORA":
        raise AdapterConfigError(f"Unsupported peft_type: {config_dict['peft_type']}")
    if config_dict["r"] <= 0:
        raise AdapterConfigError(f"Invalid rank r: {config_dict['r']}")
    return True


def diagnose_and_repair_config(config_dict):
    if not isinstance(config_dict, dict):
        return dict(DEFAULTS), ["Config was not a dict; reinitialized from defaults"]
    repaired = dict(config_dict)
    issues = []
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in repaired or repaired[field] is None:
            repaired[field] = DEFAULTS[field]
            issues.append(f"Missing field '{field}' replaced with default {DEFAULTS[field]}")
        elif not isinstance(repaired[field], expected_type):
            repaired[field] = DEFAULTS[field]
            issues.append(f"Invalid type for '{field}' replaced with default {DEFAULTS[field]}")
    if str(repaired.get("peft_type", "")).upper() != "LORA":
        issues.append(f"Unsupported peft_type '{repaired.get('peft_type')}' set to 'LORA'")
        repaired["peft_type"] = "LORA"
    if isinstance(repaired.get("r"), int) and repaired["r"] <= 0:
        issues.append(f"Non-positive rank {repaired['r']} set to default {DEFAULTS['r']}")
        repaired["r"] = DEFAULTS["r"]
    return repaired, issues
