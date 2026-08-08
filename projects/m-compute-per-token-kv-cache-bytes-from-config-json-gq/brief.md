Our team is rolling out an upgraded 8B parameter model using Grouped-Query Attention (GQA). When deploying via vLLM, we are trying to maximize the concurrent context window. However, the serving engine keeps crashing on startup with `ValueError: No available memory for the KV cache` or OOMing unpredictably when we set `max_model_len` manually based on crude math.

We need a deterministic way to compute exactly how many bytes the KV cache requires per token from the model's `config.json`, factoring in the GQA properties where `num_key_value_heads` is smaller than `num_attention_heads`. Using this, we need to replicate vLLM's startup logic to predict the exact `num_gpu_blocks` it allocates for a given VRAM budget and GPU utilization fraction.

Finally, we need a solver that takes a VRAM size and model config, and outputs the absolute maximum model length (in tokens) we can safely configure without crashing, keeping in mind that vLLM allocates memory in discrete blocks of tokens.
