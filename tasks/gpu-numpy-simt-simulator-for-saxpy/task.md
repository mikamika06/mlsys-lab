## Context

SAXPY (Single-precision **A** times **X** **P**lus **Y**) computes
$y \leftarrow a \cdot x + y$ element-wise for vectors $x, y \in \mathbb{R}^N$
and scalar $a$. Per element it is $O(1)$ arithmetic against three
global-memory operands (two reads, one write), making it the textbook
example of a memory-bandwidth-bound GPU kernel.

On a GPU, threads execute in a **SIMT** (Single Instruction, Multiple
Thread) model: one kernel body, launched across a grid of thread blocks,
every thread running the same code with its own `threadIdx`/`blockIdx`.
When the 32 threads of a warp touch **consecutive** addresses, the hardware
**coalesces** them into a single memory transaction; scattered or strided
access multiplies the transaction count and kills throughput.

## Task

Implement

```cuda
__global__ void saxpy_kernel(float* y, const float* x, int n, float a);
```

Each thread `i = blockIdx.x * blockDim.x + threadIdx.x` should compute
exactly one element:

$$ y_i \leftarrow a \cdot x_i + y_i $$

guarded by `i < n`. Thread `i` must touch address `i` of `x` and address
`i` of `y` — **coalesced**, one element per thread, no stride.

## Example

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    y[i] = a * x[i] + y[i];
}
```

## What the gate checks

The grader launches your kernel on the deterministic software GPU
(`arena.cuda_sim.GPU`) with $N=256$ elements, block size 64, and compares
the resulting `y` against the numpy reference $a \cdot x + y$:

$$ \mathrm{max\_abs\_err} = \max_i |y_i^{\text{kernel}} - y_i^{\text{ref}}| \le 10^{-9} $$

and also gates on the simulator's real transaction count:

$$ \mathrm{transactions} \le 30 $$

A correctly coalesced kernel (one element per thread, contiguous addresses)
measures 24 transactions for this configuration (8 warps × 3 coalesced
accesses each). A scattered or strided access pattern — even one that still
computes the right values — drives that number far higher (a
`(i * 7) % n`-style permutation hits 168 in the same configuration), so
"correct but not coalesced" still fails the gate.
