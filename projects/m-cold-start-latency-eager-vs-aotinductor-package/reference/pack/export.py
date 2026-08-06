def check_export_compatibility(config):
    return bool(config.get("supports_export", False))
