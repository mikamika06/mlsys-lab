def get_feature_matrix():
    return {
        "vllm": {
            "continuous_batching": True,
            "prefix_caching": True,
            "speculative_decoding": True,
            "tensor_parallelism": True,
            "quantized_weights": True,
            "multi_node": True
        },
        "sglang": {
            "continuous_batching": True,
            "prefix_caching": True,
            "speculative_decoding": True,
            "tensor_parallelism": True,
            "quantized_weights": True,
            "multi_node": True
        },
        "ollama": {
            "continuous_batching": True,
            "prefix_caching": False,
            "speculative_decoding": False,
            "tensor_parallelism": False,
            "quantized_weights": True,
            "multi_node": False
        },
        "tensorrt-llm": {
            "continuous_batching": True,
            "prefix_caching": True,
            "speculative_decoding": True,
            "tensor_parallelism": True,
            "quantized_weights": True,
            "multi_node": True
        }
    }
