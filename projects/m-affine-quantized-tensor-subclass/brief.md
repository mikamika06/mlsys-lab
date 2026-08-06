**Title:** Model generation producing garbage on Edge, throughput poor on Server

**Context:**
Our deployment fleet has grown to include everything from memory-constrained edge devices to high-throughput inference servers, plus a few nodes dedicated to continuous fine-tuning. We recently switched our weight tensors to an ad-hoc 4-bit integer format to save memory, but we're seeing completely divergent behavior across platforms.

**Symptoms:**
1. On our edge devices, the generated text looks like gibberish. The activations and weights have high variance, and our current uniform symmetric 4-bit compression seems to be wiping out the precision of our outliers entirely.
2. The fine-tuning team is complaining that their gradient updates are diverging after just a few steps. They suspect our standard quantization isn't capturing the normal distribution of the weights effectively, causing massive accumulation of errors during backprop.
3. The server inference nodes are getting terrible throughput, and the hardware utilization is extremely low. It looks like the system is getting bottlenecked trying to align scales and offsets for every single block during large-batch inference.

**What we need:**
Please build an `AffineQuantizedTensor` subclass that properly supports group-wise affine quantization (both symmetric and asymmetric). Map our three deployment targets (`edge_device`, `fine_tuning`, `server_inference`) to appropriate configurations to resolve the capacity, distribution, and throughput mismatches. Finally, implement a tool to measure the relative error (`rel_err`) between the quantized tensor and the original FP32 weights, and add a regression test to ensure our dequantization logic correctly applies the zero-point scaling, which was the root cause of the edge device gibberish.
