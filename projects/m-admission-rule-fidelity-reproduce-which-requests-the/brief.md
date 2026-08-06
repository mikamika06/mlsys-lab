Production vLLM continuous batching instances are exhibiting unpredictable end-to-end (E2E) latency spikes and queue buildup under steady request arrival rates. Operational traces reveal two distinct anomalies across different serving workloads.

First, under high-throughput conditions, internal scheduler state drift causes requests to be admitted out of expected order compared to the strict capacity and token limits specified by the admission rule. This leads to unexpected preemptions and memory fragmentation.

Second, when long-context prefill tasks (e.g., 32k tokens) arrive alongside high-frequency short generation requests (e.g., decodes), short requests experience severe Head-of-Line (HoL) blocking, causing tail latency (p99) to blow up. Tuning `max_num_seqs` too high exacerbates prefill stall, whereas setting it too low underutilizes GPUs during decode-heavy phases.

You are tasked with reproducing and diagnosing the scheduler behavior. In this unit, you will implement an admission rule simulator, build a parameter tuner that finds the optimal `max_num_seqs` cap to minimize p99 E2E latency, quantify HoL prefill-induced stalls on decodes, and write a regression test suite that catches subtle admission logic errors.
