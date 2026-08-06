Our mixed-precision quantization strategy relies on a cheap proxy—the mean absolute value of a layer's weights—to estimate how sensitive a layer is to quantization. Unfortunately, some layers are appearing robust under this proxy but show massive error when actually serving activations, causing severe perplexity degradation on long-context tasks.

The root cause is that weight-only metrics completely ignore the distribution of input activations. To solve this, we need to build a true sensitivity metric that computes the mean squared error (MSE) between the unquantized and quantized outputs given sample activations.

Your task:
1. Implement the existing cheap proxy as a baseline.
2. Implement the true sensitivity metric by simulating 4-bit weight quantization (round to the nearest 0.25 step) and computing MSE on the outputs. Use this to generate a mixed-precision recipe (assign 8 bits to layers with strictly above-median true sensitivity, and 4 bits otherwise).
3. Write a regression test that proves the new metric depends on activations, intentionally failing if a purely weight-based metric is used in its place.
