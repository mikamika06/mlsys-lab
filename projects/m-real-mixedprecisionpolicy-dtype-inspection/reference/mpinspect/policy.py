def inspect_policy(policy_obj):
    """Inspect policy dtypes."""
    return {
        "param_dtype": str(policy_obj.get("param_dtype")),
        "reduce_dtype": str(policy_obj.get("reduce_dtype")),
        "buffer_dtype": str(policy_obj.get("buffer_dtype")),
    }
