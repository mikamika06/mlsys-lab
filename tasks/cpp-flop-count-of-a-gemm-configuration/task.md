## Context

When optimizing ML or HPC workloads, it is critical to correctly measure the theoretical computational cost of a General Matrix Multiply (GEMM) before invoking BLAS. A standard GEMM computes $C = \alpha A B + \beta C$, where $A$ is $M \times K$, $B$ is $K \times N$, and $C$ is $M \times N$.

`GemmConfig` (declared in `sol.hpp`) is a real struct:

```cpp
struct GemmConfig {
    int M;
    int N;
    int K;
    double alpha;
    double beta;
};
```

## Task

Implement `gemmFlops(const GemmConfig& cfg)`, returning the exact FLOP count under this strict counting model, for each of the $M \times N$ output elements:

- The dot product takes $K$ multiplications and $K - 1$ additions.
- The dot product result is multiplied by $\alpha$ (1 multiplication).
- If $\beta \neq 0.0$: the original $C_{ij}$ is multiplied by $\beta$ (1 multiplication) and added to the $\alpha$ term (1 addition).
- If $\beta == 0.0$: BLAS optimizes this away entirely -- no $\beta$ multiplication, no extra addition. The $\alpha$ term directly overwrites $C_{ij}$.

Sum over all $M \times N$ elements and return the total as a `long long`.

## Example

```cpp
// M=10, N=10, K=10, beta=0.0
// per element: 10 mults + 9 adds + 1 alpha mult = 20 FLOPs
// total: 10 * 10 * 20 = 2000
```

## What the gate checks

`main.cpp` builds five real `GemmConfig` instances (covering `beta == 0.0`, `beta != 0.0`, `beta < 0.0`, and `K == 1`) and prints `gemmFlops(cfg)` for each, plus the real `sizeof(GemmConfig)`. Your printed output is compared byte-for-byte against `ref.cpp`, compiled and run the same way: `exact_match == 1.0`. Forgetting the `beta == 0.0` special case (always adding the +2 beta term) or off-by-one errors in the dot-product count both throw off every scenario that touches them.
