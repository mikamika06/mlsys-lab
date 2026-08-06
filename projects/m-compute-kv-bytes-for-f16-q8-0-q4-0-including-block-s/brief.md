# INCIDENT REPORT: KV Cache Memory Mismatch and Degradation

During production deployment of long-context local runners, our serving nodes encountered repeated Out-Of-Memory (OOM) exceptions and unexplained memory allocation spikes when processing concurrent multi-user sessions with FP16 KV caching.

Attempting to reduce memory footprints by switching to 8-bit (`q8_0`) and 4-bit (`q4_0`) block-quantized KV cache formats introduced two critical operational issues:
1. The memory allocation planner's static byte estimates consistently deviated from the actual resident memory consumed by quantized cache buffers. Node schedulers miscalculated available headroom and admitted tasks that triggered system swaps.
2. Downstream task quality reports indicated accuracy drops on long-context retrieval tasks when running under 4-bit KV quantization, but team members lacked exact quantitative metrics measuring retrieval degradation against FP16 and FP32 baselines.

To resolve these failures, we require a standardized accounting and evaluation toolkit `kvquant` that:
* Computes exact theoretical KV cache byte sizes for `f16`, `q8_0`, and `q4_0` formats, explicitly accounting for per-block 16-bit scale overhead.
* Measures actual resident allocation memory across backing buffers for all three formats.
* Provides a reproducible needle-in-haystack benchmark harness to quantify accuracy, cosine similarity, and relative L2 error across quantized KV representations.
