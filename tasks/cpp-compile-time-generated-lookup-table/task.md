## Context

A C++ compiler can generate lookup tables (LUTs) at compile time using
`constexpr` and `std::array`. When such an array of structs is compiled,
the compiler lays out each element's memory according to the ABI's
alignment rules, inserting padding between fields (and sometimes at the
end of the struct) so every field lands on an address matching its size.

```cpp
struct LutEntry {
    char  index;
    short doubled;
    int   squared;
};
```

`short` needs 2-byte alignment and `int` needs 4-byte alignment, so a
`LutEntry` is not simply `1 + 2 + 4 = 7` bytes: the compiler inserts a
padding byte after `index` so `doubled` lands on an even address, giving
`sizeof(LutEntry) == 8`. This is exactly the layout a `constexpr
std::array<LutEntry, N>` table generator has to reproduce byte-for-byte
when it serializes its output.

## Task

Implement, in `solve.cpp`,

```cpp
void generate_lut_bytes(int n, uint8_t* out, int out_len);
```

`out` is a caller-owned buffer of exactly `out_len == n * sizeof(LutEntry)`
bytes. For each `i` in `[0, n)`, write one `LutEntry` at `out`'s real
offset `i * sizeof(LutEntry)`, with:

- `index` = `(char) i`
- `doubled` = `(short)(i * 2)`
- `squared` = `i * i`

including every compiler-inserted padding byte, which must be `0x00`.
The simplest correct strategy: build each entry into an actual `LutEntry`
value and copy its raw bytes into `out` — that IS the ground truth here,
`LutEntry` is a real struct compiled by the real compiler, there is no
separate hand-computed spec to match.

## Example

For `i = 1`: `index = 1`, `doubled = 2`, `squared = 1`. Its 8 bytes are
`01 00 02 00 01 00 00 00` (byte 0 = index, byte 1 = padding = `00`, bytes
2-3 = doubled little-endian, bytes 4-7 = squared little-endian). For `i =
0` every field is `0`, so its 8 bytes are all `00`.

## What the gate checks

The fixed driver (`main.cpp`) generates a fixed 7-entry LUT into a
sentinel-filled (`0xFF`) buffer sized to the real `n * sizeof(LutEntry)`,
then prints the entry size followed by every output byte as two-digit hex.
The gate is an exact string match (`exact_match == 1.0`) against the
reference's printed line: any wrong field value, wrong offset, or
un-zeroed padding byte changes the hex dump and fails the gate.
