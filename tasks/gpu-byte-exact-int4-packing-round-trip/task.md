## Context

Low-bit quantized weights (int4: values `0..15`) are normally stored two
codes per byte to halve memory traffic versus one code per byte. Packing
two nibbles `lo, hi in [0,16)` into one slot is just

$$b = \text{lo} + \text{hi} \cdot 16$$

and unpacking recovers them exactly, since `lo < 16` means it never
"spills" into the `hi` term:

$$\text{hi} = \left\lfloor \frac{b}{16} \right\rfloor \qquad \text{lo} = b - \text{hi} \cdot 16$$

This only works out to the bit if every step is exact — a kernel that
packs and immediately unpacks the same values must reproduce them
*exactly*, not approximately, or every dequant downstream reads back the
wrong code.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void pack_unpack_int4(float* roundtrip, float* packed, const float* codes, int n);
```

Thread `k = blockIdx.x*blockDim.x + threadIdx.x` owns `codes[2k]` (the low
nibble) and `codes[2k+1]` (the high nibble), guarded by `2*k+1 < n`:

1. Pack: `packed[k] = codes[2k] + codes[2k+1] * 16`.
2. Unpack that same value right back: `hi = floorf(packed[k] / 16.0f)`,
   `lo = packed[k] - hi * 16.0f`, then `roundtrip[2k] = lo`,
   `roundtrip[2k+1] = hi`.

## Example

`lo=9, hi=13`: `b = 9 + 13*16 = 217`. Unpacking: `hi = floor(217/16) =
13`, `lo = 217 - 13*16 = 9` — both nibbles recovered exactly.

## What the gate checks

`max_abs_err <= 1e-9` over both `packed[]` (against `codes[0::2] +
16*codes[1::2]`) and `roundtrip[]` (against the original `codes[]`) for a
fixed 64-code input (32 packed slots, values 0-15). Swapping which code
gets the `*16` (lo/hi reversed), using `16.0f` inconsistently between pack
and unpack, or forgetting the `floorf` (leaving `hi` fractional and every
downstream `lo` wrong), all break the round trip and fail the gate.
