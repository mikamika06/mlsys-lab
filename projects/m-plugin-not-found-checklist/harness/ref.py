TEST_CHECKLIST_CASES = [
    {
        "requested": {"name": "FlashAttentionPlugin", "version": "1", "namespace": "my_plugins", "fields": ["head_dim", "num_heads"]},
        "registered": [
            {"name": "FlashAttentionPlugin", "version": "1", "namespace": "my_plugins", "fields": ["head_dim", "num_heads", "sm_version"]}
        ],
        "expected": "EXACT_MATCH"
    },
    {
        "requested": {"name": "LayerNormPlugin", "version": "1", "namespace": "", "fields": ["eps"]},
        "registered": [
            {"name": "ConvPlugin", "version": "1", "namespace": "", "fields": ["eps"]}
        ],
        "expected": "MISSING_NAME"
    },
    {
        "requested": {"name": "GELUPlugin", "version": "2", "namespace": "default", "fields": ["fast_gelu"]},
        "registered": [
            {"name": "GELUPlugin", "version": "1", "namespace": "default", "fields": ["fast_gelu"]}
        ],
        "expected": "VERSION_MISMATCH"
    },
    {
        "requested": {"name": "RoPEPlugin", "version": "1", "namespace": "prod_ns", "fields": ["rotary_dim"]},
        "registered": [
            {"name": "RoPEPlugin", "version": "1", "namespace": "dev_ns", "fields": ["rotary_dim"]}
        ],
        "expected": "NAMESPACE_MISMATCH"
    },
    {
        "requested": {"name": "SwiGLUPlugin", "version": "1", "namespace": "", "fields": ["dim", "gate_bias"]},
        "registered": [
            {"name": "SwiGLUPlugin", "version": "1", "namespace": "", "fields": ["dim"]}
        ],
        "expected": "FIELD_MISMATCH"
    }
]

TEST_SERIALIZE_PAYLOADS = [
    {"eps": 0.00001, "hidden_size": 1024, "axes": [1, 2, 3], "activation": "gelu"},
    {"scale": 2.5, "max_iters": 100, "strides": [1, 1], "name": "custom_node"},
    {"beta": 0.5, "zero_point": 0, "padding": [0, 0, 0, 0], "type": "int8_quant"}
]

TEST_DECISION_CASES = [
    {
        "node": {"op_type": "Conv", "has_custom_kernel": False, "decomposable_to_native": False},
        "trt_native": {"Conv", "MatMul"},
        "plugins": ["ConvPlugin"],
        "constraints": {"allow_plugin": True, "perf_critical": True},
        "expected": "NATIVE"
    },
    {
        "node": {"op_type": "GroupNorm", "has_custom_kernel": True, "decomposable_to_native": True},
        "trt_native": {"Conv", "MatMul"},
        "plugins": ["GroupNorm"],
        "constraints": {"allow_plugin": True, "perf_critical": True},
        "expected": "PLUGIN"
    },
    {
        "node": {"op_type": "CustomRMSNorm", "has_custom_kernel": False, "decomposable_to_native": True},
        "trt_native": {"Add", "Mul", "ReduceMean"},
        "plugins": [],
        "constraints": {"allow_plugin": True, "perf_critical": False},
        "expected": "REWRITE"
    },
    {
        "node": {"op_type": "UnsupportedOp", "has_custom_kernel": False, "decomposable_to_native": False},
        "trt_native": {"Add", "Mul"},
        "plugins": [],
        "constraints": {"allow_plugin": False, "perf_critical": False},
        "expected": "FALLBACK"
    }
]
