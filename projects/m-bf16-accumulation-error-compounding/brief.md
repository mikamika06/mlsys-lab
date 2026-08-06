# Ticket: Loss Spike and Numerical Instability in FP16/BF16 Distributed Training

## Symptom
During recent large-scale pre-training runs across distributed nodes, our monitoring pipeline logged sudden loss spikes followed by `NaN` gradients. Analysis of intermediate checkpoints revealed two distinct numerical failure modes occurring in production:

1. **Precision Degradation in Gradient Accumulation**: When accumulating small gradients over large batch sizes using BF16 native tensors without standard FP32 precision buffers, trailing bits are dropped. This causes catastrophic truncation where gradient updates stall or accumulate compounding rounding errors.
2. **Non-Deterministic Gradient Reductions**: Distributed reduction passes produce bitwise non-deterministic results across identical training runs despite fixed global seeds. This makes reproducing exact loss-spike events impossible during local debugging.

## Requirements
To address these issues, you must implement a robust numeric training utility package in `bfacc/`:

1. **`bfacc/accumulation.py`**: Implement a numerically stable accumulation engine that tracks gradient updates using a high-precision FP32 shadow master buffer alongside native BF16 tensors. You must compute relative error bounds between naive BF16 accumulation and FP32-compensated accumulation to diagnose when loss spikes originate from catastrophic truncation.
2. **`bfacc/repro.py`**: Implement a deterministic reduction wrapper and loss-spike diagnostic utility. The reduction routine must enforce bit-exact reproducibility across independent execution paths by guaranteeing deterministic FP32 floating-point reduction order across split tensor chunks.
3. **`tests/test_regression.py`**: Write a comprehensive regression suite that validates your numeric stability tools and catches uncompensated BF16 accumulation bugs.

The harness will grade your solution based on relative error metrics, accurate spike detection, and test suite fault sensitivity.
