We are serving a new transformer model via TensorRT in weakly-typed fp16 mode (`--fp16`). Unfortunately, we are seeing `NaN` values propagating to the final output under certain workloads, completely breaking inference. Our offline validation in pure fp32 does not exhibit this issue, so it is strictly a precision clipping problem.

We suspect one of the layers—likely an exponential activation inside the custom attention block—is overflowing the `float16` maximum value (~65504), producing `Inf` which later becomes `NaN` during normalization when we divide by `Inf`.

We need you to build a mini diagnostic engine and graph transformer:
1. Write a forward pass emulator in `engine.py` that tracks the dtype of the tensors layer by layer, matching how a weakly-typed TensorRT build would execute. It must identify and isolate the exact layer name that first produces a non-finite value (`Inf` or `NaN`).
2. Write a graph transformer in `transform.py` that injects cast nodes to force an `fp32` island around the offending operations.

By inserting a `cast` to `float32` before the overflow and a `cast` back to `float16` after normalization (which brings values back to a safe `[0, 1]` range), we can salvage the `--fp16` performance for the rest of the network without losing accuracy.
