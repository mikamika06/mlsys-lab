## Context

A set-associative cache splits every byte address into three fields: an **offset** within a cache line, a **set index** selecting which set the line can live in, and a **tag** that identifies which of the (many) lines mapping to that set is actually cached. For a cache with `line_bytes` bytes per line and `sets` sets:

$$\text{line\_index} = \left\lfloor \frac{\text{addr}}{\text{line\_bytes}} \right\rfloor, \qquad
\text{offset} = \text{addr} \bmod \text{line\_bytes}, \qquad
\text{set\_index} = \text{line\_index} \bmod \text{sets}, \qquad
\text{tag} = \left\lfloor \frac{\text{line\_index}}{\text{sets}} \right\rfloor.$$

`ways` (lines per set) does not affect this decomposition — it only affects how many *different* tags can simultaneously occupy one set before a conflict eviction happens.

## Task

Implement `AddrDecomp decompose_address(unsigned long addr, int line_bytes, int sets, int ways)`, returning `{tag, set_index, offset}` computed exactly as above.

## Example

For `line_bytes = 64`, `sets = 64` (a 32 KiB, 8-way cache with `ways = 8`), address `4096`: `line_index = 4096 / 64 = 64`, `offset = 4096 % 64 = 0`, `set_index = 64 % 64 = 0`, `tag = 64 / 64 = 1`.

## What the gate checks

`main.cpp` runs `decompose_address` over four fixed cache configurations (8-way 32 KiB, 4-way 16 KiB, a 1-set fully-associative-like layout, and a direct-mapped layout) crossed with thirteen fixed addresses — including boundary values right at a line edge and addresses large enough to need 64-bit arithmetic — and prints every `(tag, set_index, offset)` triple. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's. Swapping the order of the modulo/divide steps, or using `sets` where `line_bytes` belongs, produces triples that are individually plausible but wrong for most of the fixture.
