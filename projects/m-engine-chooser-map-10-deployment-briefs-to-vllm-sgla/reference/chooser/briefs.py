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
