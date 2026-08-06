# Latency Spikes and Compilation Failures with Custom Attention Operator

We are observing severe latency spikes and compilation failures when running our sequence generation pipeline under `torch.compile`. The pipeline integrates a custom high-performance attention kernel (`run_attention`) that executes as expected during eager evaluation. However, as soon as `torch.compile` is invoked on the enclosing transformer block, the runtime triggers multiple graph breaks and throws tracing errors during the AOTAutograd capture phase.

In our production logs, TorchDynamo reports that it cannot trace through the custom attention call. When attempting full-graph compilation (`fullgraph=True`), execution fails entirely, citing missing operator schemas and an inability to perform shape inference on abstract fake tensors. Consequently, the compiler falls back to eager Python dispatches around the attention block, destroying fusion opportunities across adjacent projection layers and adding significant runtime dispatch overhead.

We need to encapsulate our attention implementation inside a compile-safe PyTorch custom operator wrapper. The wrapper must register the operator schema under the `custom_flash` library namespace (`custom_flash::flash_attn`), implement an abstract fake shape kernel using `register_fake`, and guarantee full-graph compilation without graph breaks.
