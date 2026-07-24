## Context

Modern deep learning hardware uses mixed-precision matrix multiplications to maximize throughput. Two common formats are FP16 and BF16 (Brain Float). 

FP16 has 5 exponent bits and 10 mantissa bits. It provides decent precision but a very narrow dynamic range, maxing out at roughly $65,504$. 
BF16 trades mantissa bits for exponent bits (8 exponent bits, 7 mantissa bits), exactly matching the dynamic range of FP32.

When accumulating large dot products, FP16 easily overflows if both the inputs and the output are strictly FP16 ("full FP16"). To prevent overflow and catastrophic precision loss, modern architectures (like NVIDIA's Tensor Cores) perform "mixed precision": they take BF16 inputs, compute the exact product, and accumulate the sum in a wider FP32 register.

## Task

Write `compare_matmuls(A, B)`:

```python
import numpy as np

def compare_matmuls(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    ...
```

Given two FP32 matrices `A` and `B`, you must emulate and compare two matrix multiplication strategies against the true FP32 matrix multiplication $Y_{\text{true}} = A B$:

1. **BF16 with FP32 Accumulation**: 
   - Convert $A$ and $B$ to emulated BF16. Since NumPy lacks native BF16, emulate it by casting the FP32 values to `uint32`, adding `0x7FFF + ((x >> 16) & 1)` to round to nearest even, masking out the bottom 16 bits with `0xFFFF0000`, and casting back to `float32`.
   - Multiply the emulated matrices using standard FP32 accumulation (`np.matmul` on the `float32` arrays) to get $Y_{\text{bf16}}$.
2. **Full FP16**:
   - Convert $A$ and $B$ to `np.float16`.
   - Multiply them using `np.matmul(..., dtype=np.float16)` or simply `np.matmul(A_fp16, B_fp16)` which returns `float16`.
   - Convert the result back to `float32` to get $Y_{\text{fp16}}$.

Suppress overflow/invalid warnings during the FP16 matmul, as large inputs will intentionally cause overflows.

Return the maximum absolute error for both methods compared to $Y_{\text{true}}$:
`(err_bf16, err_fp16)`

## Example

```python
import numpy as np

# With large values, FP16 overflows while BF16 handles the dynamic range.
A = np.random.randn(100, 100).astype(np.float32) * 50
B = np.random.randn(100, 100).astype(np.float32) * 50

err_bf16, err_fp16 = compare_matmuls(A, B)
# err_fp16 will be `inf` (or `nan`) due to overflow!
# err_bf16 will be a finite number like 257.2
```

## What the gate checks

The grader uses random matrices with values large enough to trigger catastrophic FP16 overflow during multiplication. It tests whether the BF16 method successfully yields a bounded error while the FP16 method fails (`err_bf16 < err_fp16` or `err_fp16` is `nan`), and it verifies your error calculations exactly match the reference.
