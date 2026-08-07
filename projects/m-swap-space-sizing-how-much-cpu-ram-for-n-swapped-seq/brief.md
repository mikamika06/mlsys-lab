# Symptom: High preemption latencies and Out-of-Memory crashes on host CPU during peak load

Production vLLM serving clusters are experiencing catastrophic performance degradations under heavy concurrent prompt workloads. When GPU memory pressure spikes, the KV-cache manager attempts to preempt sequences by swapping their KV-cache blocks from GPU VRAM to CPU RAM.

Instead of smoothly offloading cached tokens, nodes frequently crash with Host Out-Of-Memory (`OOMKilled`) errors or stall for seconds when allocating CPU swap space on the fly. Preliminary operational logs indicate that the CPU swap space was configured using static rules of thumb that ignore sequence-specific head dimensions, layer counts, precision byte sizes, and dynamic block allocation overheads.

Your goal is to build an exact host CPU RAM swap-space sizing model for $N$ swapped sequences. You will write utilities to compute precise CPU allocation bounds per swapped sequence, model dynamic memory reservation trajectories under bursty preemptions, and implement a regression test suite that validates swap bounds against illegal memory truncation.
