## Context

In high-performance C++ (AVX-512 loads, cache-line-tuned data structures), buffers
are often manually aligned to a 64-byte boundary with `alignas(64)`. Even once the
buffer itself is aligned, code that receives a raw, not-necessarily-aligned address
(say, from a sub-allocator or a foreign API) must compute where inside that buffer
the next 64-byte-aligned load path begins, before it is safe to issue an aligned
write or an aligned SIMD load.

The formula to round an integer address $A$ up to an alignment $N$ (a power of 2)
is:
$$ \mathrm{Aligned}(A, N) = (A + N - 1) \ \& \ \sim(N - 1) $$

## Task

Implement

```cpp
uint64_t fill_aligned_buffer(unsigned char* storage, uint64_t base_address,
                              const float* data, int n);
```

`storage` is a real buffer the driver has already declared `alignas(64)`, so
`storage[0]` itself sits on a 64-byte boundary. `base_address` is a *synthetic*
hypothetical address (not the real address of `storage`) whose low bits are not
necessarily a multiple of 64 — it models "wherever a real allocator happened to
hand you memory".

You must:
1. Find the first address `aligned` with `aligned >= base_address` and
   `aligned % 64 == 0`, using the formula above with $N = 64$.
2. Compute `offset = aligned - base_address`. Because `storage[0]` is itself
   64-aligned, `storage + offset` is the real, in-buffer byte position that lands
   on that same 64-byte boundary.
3. Copy the `n` floats from `data` into `storage[offset .. offset + 4*n)` as raw
   bytes (a plain `memcpy` of `float`s already is native little-endian on this
   platform).
4. Return `aligned`.

Every other byte of `storage` must stay untouched — the driver zero-initialises it
before calling you.

## Example

```
storage: 256 zero bytes, alignas(64)
base_address = 1000        // 1000 % 64 == 40, not aligned
data = {3.14, 2.71}

// 1024 is the next multiple of 64
// offset = 1024 - 1000 = 24
// -> writes 3.14, 2.71 as bytes at storage[24..32)
// -> returns 1024
```

## What the gate checks

The driver calls `fill_aligned_buffer` with a fixed `base_address = 1000` and 5
fixed floats, then prints the returned `aligned` address, `aligned % 64`, and
every one of the 256 buffer bytes. The grader compiles `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference's printed output — meaning every byte of the buffer,
including the ones that must stay zero outside `[offset, offset+20)`, and the
returned address, must match bit-for-bit. Returning an address that merely
*happens* to be a multiple of 64 (e.g. `0`) is not enough: the packed float bytes
must also land at the exact computed offset, or the byte-exact comparison fails.
