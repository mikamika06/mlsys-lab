def decide_op_strategy(node_spec, trt_native_ops, available_plugins, constraints):
    op_type = node_spec.get("op_type")
    has_custom = node_spec.get("has_custom_kernel", False)
    decomposable = node_spec.get("decomposable_to_native", False)
    allow_plugin = constraints.get("allow_plugin", True)
    perf_critical = constraints.get("perf_critical", False)

    if op_type in trt_native_ops:
        return "NATIVE"

    if allow_plugin and op_type in available_plugins:
        if perf_critical or has_custom or not decomposable:
            return "PLUGIN"

    if decomposable:
        return "REWRITE"

    if allow_plugin and op_type in available_plugins:
        return "PLUGIN"

    return "FALLBACK"
