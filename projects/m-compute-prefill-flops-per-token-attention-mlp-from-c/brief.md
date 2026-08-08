# Prefill & Decode Roofline Performance Model

Serving LLMs efficiently requires understanding the fundamental arithmetic and memory bounds across prefill and decode stages. While prefill is typically compute-bound (dominated by matrix multiplications), decode is heavily memory-bandwidth bound (dominated by loading weight parameters and streaming key-value cache entries from high-bandwidth memory).

You are tasked with building a lightweight roofline performance analyzer for LLM serving engines. The engine needs exact mathematical models for:
1. Prefill FLOPs per token across attention projections and MLP layers using model parameters from a standard HuggingFace `config.json`.
2. Step-wise memory transfer volume (bytes streamed) during the decode phase for arbitrary batch sizes and sequence context lengths.
3. A hardware roofline performance predictor estimating generation throughput (`tokens/s`) and operational intensity for given GPU hardware specs (HBM bandwidth and Peak TFLOPs).

Your task is to implement the roofline analytical functions in `roofline/flops.py`, `roofline/memory.py`, and `roofline/predictor.py`, along with a robust regression test suite in `tests/test_regression.py`.
