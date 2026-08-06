def detect_truncation(imatrix_data, expected_layers):
    """Detect if an imatrix run is truncated."""
    if not isinstance(imatrix_data, dict):
        return True
    layers = imatrix_data.get("layers", [])
    if len(layers) < expected_layers:
        return True
    for l in layers:
        if "data" not in l or not l["data"]:
            return True
    return False
