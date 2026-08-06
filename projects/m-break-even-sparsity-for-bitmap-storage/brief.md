The mobile deployment team has opened a high-priority ticket regarding the export pipeline for our edge models. We use a bitmap-based storage format for sparse weights, meaning we store a 1-bit boolean mask for every weight, followed by a packed array of only the non-zero weight values.

Last week, an engineer attempted to compress our INT8 quantized model by applying magnitude pruning at 10% sparsity. However, when they exported the resulting `mlpackage`, they discovered the file size actually *increased* compared to the unpruned dense model. This caused the edge devices to reject the update due to storage limits. They suspect there might be a fundamental mathematical misunderstanding about when bitmap-based sparsity actually saves space.

Additionally, the accuracy drop was much larger than expected. The team suspects our pruning function is applying masks independently per layer instead of calculating a global threshold across all weights, which would allow critical layers to remain dense while heavily pruning less important ones. 

We need to implement tools to analyze this: a function to compute the exact break-even sparsity point, a theoretical size calculator, and accurate per-layer versus global mask generators. Finally, we need a regression test that strictly enforces the correct theoretical size logic, as forgetting the bitmap overhead is a recurring mistake.
