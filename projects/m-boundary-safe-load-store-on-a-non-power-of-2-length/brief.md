# Boundary-safe load/store on a non-power-of-2 length

A Triton kernel servicing non-power-of-2 vector lengths produces corrupted outputs or crashes under Triton's interpreter mode. Downstream consumers report missing boundary checks when array dimensions do not align cleanly with standard tile power-of-2 bounds. Unmasked operations at tile edges appear to overwrite memory out-of-bounds or trigger invalid access exceptions.

You need to establish proper masking semantics for non-power-of-2 lengths across vector operations.

Your goals for this exercise:
1. Implement Triton kernels and helper routines that safely perform memory loads and stores when the total vector length `N` is not a power of 2, masking out out-of-bounds offsets correctly.
2. Implement an interpreter validation helper that captures unmasked out-of-bounds memory stores during execution, catching real Triton interpreter access exceptions.
3. Quantify memory efficiency by calculating the wasted-lane fraction given vector lengths and candidate `BLOCK_SIZE` parameters, and write regression tests that catch unmasked memory writes.
