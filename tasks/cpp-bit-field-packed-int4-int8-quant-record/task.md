## Context

When optimizing neural networks for inference, weights are often quantized
to 4-bit integers (int4) and packed together with their scaling factor into
a memory-efficient record:

```cpp
struct QuantBlock {
    int8_t  scale;         // 8-bit signed scale
    int32_t zero_point;    // 32-bit signed zero point
    uint8_t weights[16];   // 32 packed int4 weights, 2 per byte
};
```

Under the LP64 ABI, `int32_t zero_point` needs 4-byte alignment, but it
follows a 1-byte `scale`. The compiler inserts **padding** bytes between
them so `zero_point` lands on a 4-byte boundary — the struct is not simply
its fields laid end to end.

## Task

Implement, in `solve.cpp`,

```cpp
void pack_quant_block(int8_t scale, int32_t zero_point, const int weights[32],
                       uint8_t* out, int out_len);
```

`out` is a caller-owned buffer of exactly `out_len == sizeof(QuantBlock)`
bytes. You must:

1. Pack the 32 entries of `weights` (each in `[0, 15]`) into 16 bytes: for
   `i` in `[0, 16)`, `weights[2*i]` goes in the **low** nibble and
   `weights[2*i+1]` goes in the **high** nibble of packed byte `i`.
2. Write `scale`, then `zero_point`, then the 16 packed weight bytes into
   `out` at `QuantBlock`'s real field offsets — including whatever padding
   the compiler actually inserts between `scale` and `zero_point` — in
   little-endian byte order.

The simplest correct strategy: write the fields into an actual
`QuantBlock` value and copy its raw bytes into `out`. That IS the ground
truth here — `QuantBlock` is a real struct compiled by the real compiler,
there is no separate hand-computed spec to match.

## Example

For `scale = 10`, `zero_point = -3`, and `weights[0] = 3, weights[1] = 8,
...`, the first packed weight byte is `(8 << 4) | 3 = 0x83`, and it sits at
whatever offset follows `scale`'s padding bytes and `zero_point`'s 4 bytes
on this compiler (offset 8 for the layout above). The full output is
`sizeof(QuantBlock)` bytes: `scale` at offset 0, padding, `zero_point` at
offset 4, then the 16 packed weight bytes.

## What the gate checks

The fixed driver (`main.cpp`) builds a deterministic `scale`, `zero_point`,
and 32-entry `weights` array, calls `pack_quant_block` into a
sentinel-filled (`0xFF`) buffer sized to the real `sizeof(QuantBlock)`, and
prints the struct size followed by every output byte as two-digit hex. The
gate is an exact string match (`exact_match == 1.0`) against the
reference's printed line: any wrong nibble packing, wrong field offset, or
leftover sentinel byte (padding not overwritten, or `out_len` not fully
used) changes the hex dump and fails the gate.
