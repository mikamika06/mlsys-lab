Low-precision numerics emulation and quantization mismatches in evaluation pipeline

Our offline evaluation pipeline for custom low-precision hardware ops produces subtle numerical drift when comparing Python NumPy simulations against hardware trace logs. In particular, models executing float32 to bfloat16 bit conversions exhibit discrepancies in tie-breaking cases when compared to hardware expectations. Simple truncation or float shifts in NumPy produce invalid least-significant bits for values that sit midway between representable bfloat16 values.

Additionally, casting float32 arrays to float16 during dynamic range stress testing retains non-zero subnormal values in underflow regions where the target execution engine is configured to flush subnormals to zero. Finally, float16 and bfloat16 range checks and ULP metrics in our model diagnostic suite report invalid maximum representable finite limits and inaccurate ULP steps across normalized and subnormal ranges.

We need a clean, vectorized Python library in `bf16num` that provides:
1. Bit-exact FP32 to BF16 conversion with IEEE 754 round-to-nearest-even tie breaking.
2. FP16 subnormal detection and configurable subnormal flushing during downcasting.
3. Exact max finite value lookups and dynamic ULP computation for FP32, FP16, and BF16 dtypes.
4. A test suite in `tests/test_regression.py` validating these numerical behavior invariants.
