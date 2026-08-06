def resolve_actual_device(compiled_model_properties, target_hint):
    """Determine actual selected execution device based on runtime properties."""
    raise NotImplementedError


def inspect_auto_allocations(deployment_targets):
    """Inspect actual device selections for a list of deployment target specs."""
    raise NotImplementedError
