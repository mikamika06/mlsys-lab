Ticket #TRT-8941: Model conversion pipeline sporadically fails numerical validation during TensorRT layer-wise debug-reduce runs.

Engineers running automated Polygraphy bisection reduced workflows on multi-stage ONNX execution graphs report that the isolation tool is incorrectly flagging early layers as failing. In several benchmark models, tiny FP16 denormal accumulation and subnormal floating-point noise trigger premature divergence signals during intermediate layer comparisons. This causes the bisection engine to halt prematurely or spend excessive log iterations scanning benign layers.

Furthermore, when structural bug isolation runs, genuine activation mismatches (such as layer weight corruption or transposed output dimensions) are sometimes lumped together with subnormal precision noise, masking severe graph bugs as precision drift.

We need a dedicated bisection debug-reduce helper and sanitization layer. The utility must perform binary search over step indices to minimize evaluation steps, sanitize floating-point noise (specifically zeroing out subnormals and handling floating-point representation artifacts), and accurately classify tensor comparisons as exact matches, ignorable floating-point noise, or genuine bugs. A test suite must be established to prevent regression when subnormal thresholds or bisection logic are updated.
