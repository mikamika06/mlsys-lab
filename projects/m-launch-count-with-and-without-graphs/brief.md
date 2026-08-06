Production latency spikes under variable batch sizes in PyTorch models are often traced to launch overheads and repeated kernel dispatch. While CUDA Graphs can drastically cut launch overheads, naively re-capturing graphs for every input shape invalidates performance gains. Conversely, avoiding graph capture entirely retains maximum launch overheads for every step.

To optimize high-throughput model dispatch, we must systematically quantify the kernel launch overheads across execution modes, model launch speedups, and implement a static-buffer harness capable of managing dynamic input tensors without invalidating graph capture boundaries.

You are tasked with implementing three modules in `launchgraph/`:
1. `launches.py`: Profile kernel launch overheads and count dispatches with and without CUDA Graphs.
2. `predict.py`: Model and predict speedups from graph capture based on step execution time and fixed overheads.
3. `harness.py`: Implement a static-buffer harness that pads dynamic inputs into fixed memory addresses to preserve captured graph calls safely.
