# Import Safetensors, Remote Blob Creation, and Requantization Benchmarking

Our team is automating the import pipeline for custom fine-tuned weights, remote blob management in Ollama-compatible servers, and GGUF quantization validation. We have received reports that our newly imported models exhibit degraded perplexity or unexpected weight distortions when converted via different quantization pipelines.

Specifically, downstream services report three key issues:
1. When importing raw `safetensors` model directories, local tooling occasionally accepts unsupported model architectures or misinterprets key layer shapes before creating GGUF artifacts.
2. Direct remote weight uploads using blob endpoints (`POST /api/blobs/:digest`) and Manifest/Model creation do not correctly track missing blob digests or payload byte offsets.
3. Quantizing models through Ollama's direct quantization pathway yields non-identical weight block norms and quantization errors compared to standard `llama-quantize` for the same target target type (e.g., `Q4_0`, `Q8_0`).

You need to implement the pipeline tools in `runner/import_tools.py`, `runner/remote_blobs.py`, and `runner/quant_compare.py`. Finally, write a suite of regression tests in `tests/test_regression.py` that verifies weight conversion invariants and catches quiet architecture misconfigurations.
