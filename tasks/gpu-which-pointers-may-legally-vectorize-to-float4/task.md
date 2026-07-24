## Context

CUDA's `float4` vector load/store instruction reads or writes **16 bytes** in a single
transaction.  For this to be legal the pointer must be **16-byte aligned** — i.e., its
byte address must satisfy

$$\text{addr} \bmod 16 = 0$$

If the alignment requirement is violated the hardware raises a misaligned-address error
(or silently reads garbage on older architectures).  Knowing which pointers are
vectorizable lets a kernel writer safely emit wide loads and cut the number of
global-memory transactions by $4\times$.

A pointer to `float` (4 bytes) is legal for `float4` if and only if:

$$\text{addr} \bmod 16 = 0$$

## Task

Implement `vectorizable_pointers(addrs: list[int]) -> list[bool]` that, given a list
of byte addresses, returns a Boolean list where `True` means the address is
**16-byte aligned** and may legally be used with `float4`.

$$\text{out}[i] = (\text{addrs}[i] \bmod 16 = 0)$$

## Example

```python
addrs = [0, 4, 16, 32, 48, 60, 64, 100]
vectorizable_pointers(addrs)
# -> [True, False, True, True, True, False, True, False]
```

## What the gate checks

`check.py` generates a list of addresses (a mix of aligned and unaligned), computes the
reference Boolean vector using $\text{addr} \bmod 16 = 0$, and checks that your output
matches exactly (`exact_match == 1.0`).
