ENGINES = ["vllm", "sglang", "ollama", "tensorrt-llm"]

BRIEFS = [
    {"id": 1, "constraint": "local desktop, consumer hardware, simple setup", "expected": "ollama"},
    {"id": 2, "constraint": "ultra-high throughput production, RadixAttention, deep tensor parallelism", "expected": "sglang"},
    {"id": 3, "constraint": "enterprise production, mature ecosystem, continuous batching, extensive OpenAI-compatible API", "expected": "vllm"},
    {"id": 4, "constraint": "NVIDIA-locked bare metal maximum possible inference speed, static tensor RT engines", "expected": "tensorrt-llm"},
    {"id": 5, "constraint": "macOS or PC local playground with minimal dependencies", "expected": "ollama"},
    {"id": 6, "constraint": "multi-node tensor and pipeline parallelism with advanced paged attention and prefix caching in vllm", "expected": "vllm"},
    {"id": 7, "constraint": "structured generation / JSON regex constraints with high concurrency using RadixAttention", "expected": "sglang"},
    {"id": 8, "constraint": "maximum throughput on NVIDIA H100 clusters compiled down to TensorRT binaries", "expected": "tensorrt-llm"},
    {"id": 9, "constraint": "standard OpenAI API drop-in replacement with LoRA adapter serving support", "expected": "vllm"},
    {"id": 10, "constraint": "local running of GGUF models on consumer laptops", "expected": "ollama"}
]

def classify_brief(brief):
    text = brief["constraint"].lower()
    if "local" in text or "desktop" in text or "macos" in text or "consumer" in text or "gguf" in text:
        return "ollama"
    if "radix" in text or "structured" in text:
        return "sglang"
    if "tensorrt" in text or "compiled" in text or "h100" in text:
        return "tensorrt-llm"
    return "vllm"

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

def estimate_memory(num_params_b, bits_weight, kv_cache_bytes):
    weight_bytes = int(num_params_b * 1e9 * bits_weight / 8.0)
    return weight_bytes + kv_cache_bytes

def compare_gguf_vs_w4a16(num_params_b, kv_cache_bytes):
    gguf_mem = estimate_memory(num_params_b, 4.5, kv_cache_bytes)
    w4a16_mem = estimate_memory(num_params_b, 4.0, kv_cache_bytes) + int(num_params_b * 1e9 * 0.5 / 8.0)
    return {
        "gguf_q4_bytes": gguf_mem,
        "w4a16_bytes": w4a16_mem,
        "diff_bytes": abs(gguf_mem - w4a16_mem)
    }
