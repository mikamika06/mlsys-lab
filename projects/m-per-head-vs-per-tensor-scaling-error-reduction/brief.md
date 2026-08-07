We are rolling out an FP8 (E4M3) KV cache to halve our memory footprint during serving. Early offline metrics looked good, but end-to-end generation is yielding repetitive loops and degraded reasoning.

When inspecting the attention matrices, we found that for certain layers, specific attention heads are outputting uniform noise or pure zeros, effectively wiping out their contribution to the forward pass. We suspect this is an outlier-driven quantization failure. Currently, we compute a single scaling factor per tensor (layer) to map the activations into the E4M3 range (max value ~448.0). If one head contains an extreme outlier, it shrinks the scale factor so aggressively that normal activations in other heads quantize to zero.

Your task is to:
1. Implement basic E4M3 quantization simulation (rounding to integers in the range [-448, 448] as a proxy for the scaling effect).
2. Compute scaling factors under two regimes: per-tensor and per-head.
3. Write an analysis function that measures the relative error introduced by quantization, and identifies the "breaking head"—the head that suffers the worst degradation when forced to share a scale with an outlier.
4. Calculate the KV cache capacity differences (including the byte overhead of the scaling factors).

If our hypothesis holds, moving to a per-head scale will save the model's accuracy while adding only a tiny memory overhead.
