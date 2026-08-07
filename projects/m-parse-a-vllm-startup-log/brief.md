# Incident Report: Production Serving Garbage Output under Tensor Parallelism

## Symptom
Our team recently deployed a quantized large language model using vLLM in a multi-GPU cluster. When running with a tensor parallel (TP) size of 1, the model serves requests accurately and responds with completely coherent, high-quality text. However, the moment we scale the tensor parallelism configuration to TP=2 or higher across multiple GPUs, the service starts generating completely unreadable garbage output—random repeating tokens, mojibake, and structural gibberish—despite successful initialization and normal log patterns during startup.

Furthermore, during routine operational audits and log parsing of our startup sequences, we noticed that certain automated deployment pipelines fail to flag illegal or unsupported tensor parallel sharding configurations for specific layer shapes and quantization parameters. This leads to silent configuration mismatches where weight dimensions, attention head counts, and tensor parallel degrees are incompatible, yet the engine boots up without raising a hard exception, manifesting later as corrupted inference outputs.

We need a robust programmatic utility to parse vLLM startup logs, validate whether a given tensor parallel sharding configuration is mathematically legal for a model architecture and quantization scheme, and accurately diagnose the root cause of garbage output when TP > 1.
