def trace_intermediates(fn, sample_inputs, autocast_dtype="float16"):
    """Trace all intermediate tensor dtypes during execution."""
    FP32_OPS = {"layernorm", "softmax", "loss", "log", "exp", "reciprocal"}
    FP16_OPS = {"matmul", "conv2d", "linear", "addmm"}
    CAST_OPS = {"cast", "to"}

    intermediates = []
    
    if isinstance(sample_inputs, dict):
        graph = sample_inputs.get("ops", [])
        inputs = sample_inputs.get("inputs", {})
    else:
        graph = fn(sample_inputs) if callable(fn) else []
        inputs = {}

    env = dict(inputs)

    for step in graph:
        op = step["op"]
        args = step["args"]
        out_name = step["out"]
        
        arg_dtypes = [env.get(a, "float32") for a in args]
        
        if op in CAST_OPS:
            out_dtype = step.get("target_dtype", "float32")
            reason = "explicit_cast"
        elif op in FP32_OPS:
            out_dtype = "float32"
            reason = "op_requires_fp32"
        elif op in FP16_OPS:
            out_dtype = autocast_dtype
            reason = "autocast_promoted"
        else:
            if any(d == "float32" for d in arg_dtypes):
                out_dtype = "float32"
                reason = "promoted_from_fp32_input"
            else:
                out_dtype = autocast_dtype
                reason = "passthrough"

        env[out_name] = out_dtype
        intermediates.append({
            "out": out_name,
            "op": op,
            "actual_dtype": out_dtype,
            "arg_dtypes": arg_dtypes,
            "reason": reason
        })

    return {"intermediates": intermediates, "env": env, "autocast_dtype": autocast_dtype}
