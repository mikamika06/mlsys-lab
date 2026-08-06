def generate_test_cases():
    return [
        {
            "inputs": {"x": "float32", "w1": "float32", "w2": "float32"},
            "ops": [
                {"op": "matmul", "args": ["x", "w1"], "out": "mm1"},
                {"op": "layernorm", "args": ["mm1"], "out": "ln1"},
                {"op": "matmul", "args": ["ln1", "w2"], "out": "mm2"},
                {"op": "softmax", "args": ["mm2"], "out": "sm1"},
                {"op": "cast", "args": ["sm1"], "target_dtype": "float32", "out": "out"}
            ]
        },
        {
            "inputs": {"a": "float16", "b": "float32"},
            "ops": [
                {"op": "add", "args": ["a", "b"], "out": "sum1"},
                {"op": "conv2d", "args": ["sum1"], "out": "c1"},
                {"op": "log", "args": ["c1"], "out": "l1"}
            ]
        }
    ]


def reference_trace(sample_inputs, autocast_dtype="float16"):
    FP32_OPS = {"layernorm", "softmax", "loss", "log", "exp", "reciprocal"}
    FP16_OPS = {"matmul", "conv2d", "linear", "addmm"}
    CAST_OPS = {"cast", "to"}

    intermediates = []
    graph = sample_inputs.get("ops", [])
    inputs = sample_inputs.get("inputs", {})
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


def reference_diagnose(trace_result):
    diagnoses = []
    for item in trace_result.get("intermediates", []):
        if item["actual_dtype"] == "float32":
            op = item["op"]
            reason = item["reason"]
            if reason == "op_requires_fp32":
                explanation = f"Operation '{op}' is registered as strictly FP32 for numerical stability."
            elif reason == "explicit_cast":
                explanation = f"Operation '{op}' has an explicit target_dtype override to float32."
            elif reason == "promoted_from_fp32_input":
                explanation = f"Operation '{op}' was promoted to float32 due to FP32 input operands."
            else:
                explanation = f"Operation '{op}' stayed in float32 due to default fallback rules."
            
            diagnoses.append({
                "out": item["out"],
                "op": op,
                "dtype": "float32",
                "reason": reason,
                "explanation": explanation
            })
    return diagnoses
