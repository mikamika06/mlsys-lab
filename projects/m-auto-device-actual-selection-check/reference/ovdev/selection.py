def resolve_actual_device(compiled_model_properties, target_hint):
    """Determine actual selected execution device based on runtime properties."""
    exec_devices = compiled_model_properties.get("EXECUTION_DEVICES", [])
    if isinstance(exec_devices, str):
        exec_devices = [exec_devices] if exec_devices else []
    
    if exec_devices:
        return exec_devices[0]

    actual = compiled_model_properties.get("EXEC_DEVICE_NAME")
    if actual:
        return actual

    available = compiled_model_properties.get("AVAILABLE_DEVICES", [])
    if target_hint == "THROUGHPUT" and "GPU" in available:
        return "GPU"
    if target_hint == "LATENCY" and "NPU" in available:
        return "NPU"

    return available[0] if available else "CPU"


def inspect_auto_allocations(deployment_targets):
    """Inspect actual device selections for a list of deployment target specs."""
    results = []
    for target in deployment_targets:
        props = target.get("properties", {})
        hint = target.get("hint", "DEFAULT")
        actual = resolve_actual_device(props, hint)
        results.append({
            "target_id": target["id"],
            "requested_device": target.get("device", "AUTO"),
            "actual_device": actual,
            "is_fallback": actual != target.get("preferred_device", actual)
        })
    return results
