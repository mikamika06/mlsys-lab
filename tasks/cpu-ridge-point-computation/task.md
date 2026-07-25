## Context

The **roofline model** characterizes a machine by two ceilings: peak
floating-point throughput $P$ (in FLOP/s) and peak memory bandwidth $B$
(in byte/s). The **ridge point** (or balance point) is the arithmetic
intensity at which the two ceilings meet:

$$I^{*} = \frac{P}{B}$$

For a kernel whose own operational intensity $I$ (FLOP/byte) satisfies
$I < I^{*}$, the kernel is **memory-bound** on this device — moving data
is the bottleneck, and extra compute is free. When $I > I^{*}$, the
kernel is **compute-bound** — arithmetic is the bottleneck, and extra
bandwidth would go unused. The ridge point is a property of the
*hardware alone*: it doesn't depend on which kernel you're about to run,
only on the two peak numbers.

## Task

Implement, in real C++:

```cpp
double ridge_point(double peak_flops, double peak_bw);
```

`peak_flops` is the device's peak FLOP/s, `peak_bw` is its peak
bytes/s. Return $I^{*} = \text{peak\_flops} / \text{peak\_bw}$, in
FLOP/byte.

## Example

`peak_flops = 16e12` (16 TFLOP/s), `peak_bw = 2e12` (2 TB/s): ridge point
`= 8.0` FLOP/byte. A kernel with arithmetic intensity `4.0` is
memory-bound on this device; one with intensity `16.0` is compute-bound.

## What the gate checks

`main.cpp` calls `ridge_point` on 5 fixed `(peak_flops, peak_bw)` device
specs spanning several orders of magnitude and prints each result at
full `double` precision. `max_abs_err <= 1e-6` compares every printed
number against the reference computation. Dividing the operands the
wrong way round (`peak_bw / peak_flops`, which gives byte/FLOP instead
of FLOP/byte) or returning a placeholder produces numbers off by many
orders of magnitude and fails on all 5 fixtures at once.
