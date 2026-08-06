# Expert Dispatch Roofline and Expert Bin-Packing

During scale testing of MoE routing across multi-GPU nodes with Expert Parallelism (EP), the dispatch and combine phases exhibit unpredictable latency spikes under non-uniform token routing distributions. Profiling reveals two distinct operational bottlenecks: network-bound communication overhead when token routing is small, and compute-bound expert execution when local expert batch sizes surge. Additionally, naive round-robin expert placement leads to severe GPU memory and compute imbalance across ranks.

You are tasked with building an operational dispatch planner and expert placement optimizer.

1. **Compute Roofline Crossover**: Analyze the MoE dispatch/combine communication overhead against local expert GEMM compute throughput. Calculate the critical token batch size threshold ($T^*$) where the system transitions between communication-bound and compute-bound regimes based on model hyperparameters and network bandwidth.
2. **Min-Max Expert Bin-Packing**: Implement an expert assignment algorithm that distributes experts across GPUs to minimize the maximum token load (max-loaded rank) given predicted token routing frequencies, respecting memory capacity bounds per rank.
3. **Regression Safeguard**: Author regression tests in `tests/test_regression.py` that verify expert dispatch roofline calculations and ensure expert placement correctly prevents rank overload under skewed token routing distributions.
