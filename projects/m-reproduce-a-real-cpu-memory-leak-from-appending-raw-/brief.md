# Ticket: Production Fine-Tuning Job Crashes Due to Unbound Memory Growth

During extended overnight fine-tuning and evaluation runs on our cluster, several cluster nodes experience severe out-of-memory terminations. The failures occur after a few thousand training steps, long past initial warmup and stable operation. Interestingly, the crash profile differs between phases: during the main training loop, host CPU RAM steadily climbs hour after hour without ever stabilizing or triggering Python's garbage collector effectively, eventually exhausting all available system memory and causing the Linux kernel's OOM killer to terminate the process.

Concurrently, monitoring dashboards reveal that GPU memory usage during evaluation phases remains elevated long after the evaluation epoch completes, preventing subsequent training batches from fitting and triggering CUDA OOM errors. Furthermore, repeated inference-style evaluation loops using KV-caches show a compounding memory footprint that grows linearly with iteration count rather than remaining bounded.

We need to isolate these memory leaks and retention issues, implement rigorous checks to quantify host memory accumulation from loss logging, verify proper activation dropping during evaluation, and enforce strict bounds on KV-cache memory dynamics to ensure stable multi-day training runs.
