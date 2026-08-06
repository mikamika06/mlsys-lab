# L2-Cache-Aware Block Scheduling in Triton

We are observing severe memory bandwidth saturation and poor L2 cache hit rates when executing matrix multiplication kernels with large grid sizes in Triton. By default, linear program-ID scheduling assigns consecutive tile computations sequentially across SMs, causing consecutive program IDs to access distant regions of the right-hand matrix $B$. This destroys temporal L2 locality across parallel tile loads.

To recover performance, Triton employs a grouped program-ID remapping technique that schedules tiles in column-major blocks of size `GROUP_SIZE_M`, forcing tiles that reuse the same blocks of $B$ to execute concurrently on nearby SMs.

Your goal is to build an analytical simulation layer and remapping utility that models grouped program-ID ordering:
1. Implement the exact Triton 2D grouped PID-remapping algorithm to compute 2D tile coordinates `(tile_id_m, tile_id_n)` from a 1D launch index `pid`.
2. Compute dynamic SRAM block-load counts under LRU/FIFO visual cache models, generalizing the Triton matmul tutorial's tile access reuse metrics for arbitrary grid sizes $(num\_pid\_m, num\_pid\_n)$.
3. Search for the optimal `GROUP_SIZE_M` parameter that minimizes total simulated tile block fetches for arbitrary GEMM grid topologies.
4. Add a safeguard regression suite that verifies remapping bounds and catches invalid grouping implementations.
