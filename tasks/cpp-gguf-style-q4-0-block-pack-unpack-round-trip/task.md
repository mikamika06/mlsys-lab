## Context

In memory-constrained environments, large language models use highly
optimized quantization formats. GGUF (from `llama.cpp`) defines several
blocked quantization schemes; the foundational one is `Q4_0`.

A `Q4_0` block compresses 32 float32 weights into exactly 18 bytes:

```c
struct block_q4_0 {
    ggml_fp16_t d;    // delta (scaling factor), IEEE-754 binary16
    uint8_t     qs[16];  // 32 packed int4 weights, 2 per byte
};
```

Under the LP64 ABI this needs no padding at all: `d` (2-byte alignment)
and `qs` (1-byte alignment) pack back-to-back into exactly 18 bytes.

The `Q4_0` quantization rule:

1. $d = \max_i(|w_i|) / 7.0$ — the block's scale. If every weight is `0`,
   $d = 0$.
2. $q_i = \operatorname{clip}(\operatorname{round}(w_i / d), -8, 7)$ for
   each of the 32 weights (all $q_i = 0$ if $d = 0$ — never divide by
   zero).
3. Each 4-bit signed value is stored biased by `+8`, giving the unsigned
   range $[0, 15]$ that fits a nibble.
4. Byte $i \in \{0, \dots, 15\}$ of `qs` packs weight $i$ into its **low**
   nibble and weight $i+16$ into its **high** nibble:
   `qs[i] = (q[i] + 8) | ((q[i + 16] + 8) << 4)`.

## Task

Implement, in `solve.cpp`, both functions declared in `sol.hpp`:

```cpp
void pack_q4_0(const float weights[32], uint8_t* out, int out_len);
void unpack_q4_0(const uint8_t* block, int block_len, float out_weights[32]);
```

`out`/`block` are exactly `sizeof(block_q4_0) == 18` bytes. Use the
**provided** `encode_fp16` / `decode_fp16` helpers (declared in `sol.hpp`,
defined in `main.cpp`) for the scale — do not hand-roll your own float16
bit-twiddling, that is not the point of this task.

`pack_q4_0`: compute `d`, quantize and bias every weight, pack the 16
nibble-pair bytes, encode `d`, and write the resulting `block_q4_0`'s raw
bytes into `out`.

`unpack_q4_0`: the inverse — decode `d`, unpack each byte's two nibbles
back to signed `q0`/`q1` (subtract `8`), and reconstruct
`out_weights[i] = q0 * d`, `out_weights[i+16] = q1 * d`.

## Example

For weights with `max(|w|) = 3.5`, `d = 0.5`. A weight `w = 3.0` quantizes
to `round(3.0 / 0.5) = 6`, clipped to stay in `[-8, 7]` (already in
range), biased to `6 + 8 = 14`. If it's weight index `0`, it lands in the
low nibble of `qs[0]`.

## What the gate checks

The fixed driver (`main.cpp`) packs two fixed weight blocks (one
mixed-magnitude, one all-zero — the `d == 0` edge case) into
sentinel-filled buffers, prints the packed bytes as hex, unpacks them
again, and prints the 32 reconstructed floats. The gate is an exact
string match (`exact_match == 1.0`) against the reference's printed
output: a wrong scale, a wrong clip/round, swapped nibbles, or a
mishandled all-zero block all change the bytes or the round-tripped
floats and fail the gate.
