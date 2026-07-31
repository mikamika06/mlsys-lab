# Nibble packing for 4-bit weights (round trip)

We're moving part of the model's weights to a 4-bit storage format — the same
idea as bitsandbytes NF4/FP4: each weight is replaced with a 4-bit code
(0..15), two codes are packed into one byte, and one float is kept per block
— `absmax` — which the code needs to be multiplied by to get back an
approximate weight.

A colleague already wrote the packer. The file comes out exactly half the
size — looks like it works. But when the weights are unpacked again to check
against the original, neighboring values sometimes "swap": where `[3, 10]`
should be, `[10, 3]` shows up instead. And on a block of odd length,
unpacking either crashes or silently drops the last element.

## What you write

`nibblepack/pack.py`:

```python
pack_nibbles(codes) -> np.ndarray[uint8]        # length = ceil(n / 2)
unpack_nibbles(packed, n) -> np.ndarray[uint8]  # length = n
```

`codes` is an array of 4-bit codes (each 0..15), of length `n` (which may be
odd). In each byte, the low nibble (bits 0-3) holds the code at the even
index (`codes[2*i]`), and the high nibble (bits 4-7) holds the code at the
odd index (`codes[2*i+1]`). If `n` is odd, the high nibble of the last byte
is zero (not garbage). `unpack_nibbles` is the exact inverse: for any `n` and
any codes, `unpack_nibbles(pack_nibbles(codes), n)` must return the same
codes in the same order.

`nibblepack/dequant.py`:

```python
dequantize_block(packed, n, absmax, codebook=CODEBOOK) -> np.ndarray[float64]
```

`CODEBOOK` is an array of 16 numbers, already in the file — no need to touch
it. Unpack the nibbles via `unpack_nibbles`, look up the matching value in
`CODEBOOK` for each code, and multiply by `absmax`. The result has length
`n`. If `absmax == 0`, the result is a block of zeros regardless of the
codes.

## How it's graded

The grader computes the reference itself — an independent implementation run
over a set of generated blocks of varying length (even and odd, including
empty and single-nibble) and varying `absmax`, including zero.

The third milestone is yours: you write a test that checks
`unpack_nibbles(pack_nibbles(codes), n) == codes` for several arrays
(preferably ones where neighboring codes differ). We swap in an
`unpack_nibbles` that mixes up the low and high nibble. Your test needs to
catch it.

```
mlsys project start m-nibble-packing-round-trip
mlsys project grade m-nibble-packing-round-trip --milestone 1
```
