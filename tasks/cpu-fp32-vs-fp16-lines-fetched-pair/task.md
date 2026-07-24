## Context

Halving element width doubles how many elements fit in a cache line,
which halves how many distinct lines a sequential scan touches (for a
large enough array). This is one of the concrete, mechanical reasons
narrower dtypes (fp16/bf16 instead of fp32) move less traffic through the
memory hierarchy for the same logical loop over the same number of
elements — independent of any reduction in arithmetic cost.

## Task

Implement

```cpp
void compare_fp32_fp16_lines(long base, int n, int line_bytes, long* out);
```

Run the same sequential scan over `n` elements starting at byte address
`base` twice: once treating elements as 4-byte fp32 (address of element
`i` is `base + i*4`), once as 2-byte fp16 (`base + i*2`). Two addresses
land in the same cache line iff `addr / line_bytes` is equal. Write the
number of distinct lines touched by the fp32 scan into `out[0]` and by
the fp16 scan into `out[1]`.

## Example

With `line_bytes = 64`, 16 fp32 elements (4 bytes) fit per line, so 100
elements touch `ceil(100/16) = 7` distinct lines. 32 fp16 elements
(2 bytes) fit per line, so the same 100 elements touch
`ceil(100/32) = 4` lines — roughly half, matching the halved element
width.

## What the gate checks

`exact_match`: the driver prints both line counts for a fixed 100-element
scan starting at a line-aligned base address. Using the wrong element
width, or the same width for both outputs, gives a count that does not
match the reference; a starter returning `0, 0` fails outright.
