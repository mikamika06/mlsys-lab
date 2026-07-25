## Context

Shared memory is divided into 32 banks, word-interleaved: word index $w$
lives in bank $w \bmod 32$. If every lane in a warp accesses a
*different* bank, all 32 accesses are serviced in one cycle. If several
lanes hit the *same* bank with *different* addresses, those accesses
serialize — an $n$-way conflict takes $n$ cycles instead of 1. There is
exactly one exception: if every one of those lanes requests the *exact
same* address (not just the same bank), hardware serves them all in one
shot — a **broadcast**, not a conflict, no matter how many lanes share it.

So the real conflict degree of a bank isn't "how many lanes touched it" —
it's "how many *distinct addresses* were requested at it." A 12-lane
broadcast to one address costs the same one cycle as a single lane; four
lanes touching four different words in the same bank cost four cycles no
matter how few lanes are involved.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void bank_conflict_probe(float* out, const float* seed, const float* idx);
```

Launched as one 32-thread block (`lane = threadIdx.x`). First,
cooperatively fill a 128-word `__shared__` buffer from `seed` (every lane
fills its strided quarter: `for (int j = lane; j < 128; j = j + 32)
smem[j] = seed[j];`), then `__syncthreads()`, then have each lane read
back exactly one shared word at offset `idx[lane]` (a fixed, given
per-lane access pattern) and write it to `out[lane]`.

## Example

Word 16 and word 48 are both bank 16 (`16 mod 32 == 48 mod 32`). If 8
lanes all request word 16, that's a broadcast — 1 distinct address, no
conflict. If instead 4 lanes request four *different* bank-16 words —
say 16, 48, 80, 112 — that's a real 4-way conflict: 4 distinct addresses
serialize onto the same bank.

## What the gate checks

`max_abs_err <= 1e-9` (every `out[lane]` must equal `seed[idx[lane]]`)
**and** `smem_waves == 8` on the fixed launch, where `idx` is: lanes 0-15
each read a distinct word (`0..15` — no conflict), lanes 16-27 all read
word 16 (a 12-lane broadcast — exempt), and lanes 28-31 read words `17,
49, 81, 113` (four distinct bank-17 words — a genuine 4-way conflict).
The simulator's `smem_waves` totals 8: the four cooperative fill passes
(conflict-free, common to any correct implementation — one warp filling
128 words 32 at a time) plus this probe's measured 4-way conflict. Reading
`out[lane]` from `idx[lane]` incorrectly, skipping the `__syncthreads()`,
or under/over-filling the shared buffer changes the printed values,
the wave count, or both.
