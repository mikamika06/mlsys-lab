# Ticket: Production Serving Engine Selection Failure and Memory Footprint Discrepancy

## Symptom Report

Our machine learning platform team is experiencing severe integration bottlenecks when deploying new large language models across diverse hardware footprints and latency-sensitive production environments. Recently, several staging deployments failed unexpectedly or suffered from sub-optimal resource allocation due to incorrect inference engine selection and unverified memory footprint calculations.

Specifically, engineering teams are struggling to systematically map specific production deployment constraints—ranging from local consumer hardware and edge setups to ultra-high throughput multi-node clusters utilizing advanced speculative decoding and RadixAttention—to the correct serving backends among vLLM, SGLang, Ollama, and TensorRT-LLM. Without a programmatic classification mapping 10 distinct deployment briefs, teams frequently misconfigure serving runtimes.

Furthermore, during architectural reviews, we discovered inconsistencies in reconstructing the official feature capability matrix across these four engines for critical capabilities such as continuous batching, prefix caching, speculative decoding, and tensor parallelism. To make matters worse, memory footprint estimations comparing quantized formats like GGUF Q4 versus W4A16 vLLM paths are currently unverified, leading to out-of-memory kernel crashes under peak concurrent request loads.

We require a robust, automated module that accurately classifies deployment briefs, validates feature capabilities against official specifications, and calculates precise memory footprint differentials with rigorous regression test safeguards.
