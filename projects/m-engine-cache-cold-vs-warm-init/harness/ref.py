CONFIGS = [
    {
        "cache_meta": {"device_id": 0, "trt_version": "8.6.1", "profiles": [{"min": [1, 3, 32, 32], "max": [1, 3, 128, 128]}]},
        "runtime_config": {"device_id": 0, "trt_version": "8.6.1", "profiles": [{"min": [1, 3, 32, 32], "max": [1, 3, 128, 128]}]},
        "expected_valid": True
    },
    {
        "cache_meta": {"device_id": 0, "trt_version": "8.6.1", "profiles": [{"min": [1, 3, 32, 32], "max": [1, 3, 128, 128]}]},
        "runtime_config": {"device_id": 1, "trt_version": "8.6.1", "profiles": [{"min": [1, 3, 32, 32], "max": [1, 3, 128, 128]}]},
        "expected_valid": False
    },
    {
        "cache_meta": {"device_id": 0, "trt_version": "8.6.1", "profiles": [{"min": [1, 3, 32, 32], "max": [1, 3, 128, 128]}]},
        "runtime_config": {"device_id": 0, "trt_version": "8.6.2", "profiles": [{"min": [1, 3, 32, 32], "max": [1, 3, 128, 128]}]},
        "expected_valid": False
    }
]

GRAPH_NODES = ["Conv_0", "Relu_1", "MaxPool_2", "Gemm_3", "Softmax_4"]
SUPPORTED_OPS = {"Conv_0", "Relu_1", "MaxPool_2", "Gemm_3", "Softmax_4"}
