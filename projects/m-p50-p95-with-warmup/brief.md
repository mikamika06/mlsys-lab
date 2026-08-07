Our team has been profiling the newly exported Edge models, and the latency numbers look completely wild in our CI and on-device dashboards.

We are getting reports that the p95 latency on-device is over 500ms, but when we trace the actual operations, nothing takes longer than 15ms. It turns out our benchmark loop just measures time across 100 inferences and calculates the percentiles. It includes the very first inference (which triggers the NPU initialization and allocations) and is highly susceptible to OS scheduling hiccups on the mobile device.

We need a standardized `benchmark` function in `measure.py` that fixes this:
1. First runs `warmup` iterations (without recording their latencies for the main distribution) to get past the cold start.
2. Then runs `iters` iterations to compute the actual p50 and p95 latencies.
3. Captures the "cold-start inflation" — the ratio of the very first warmup latency to the p50 of the measured iterations.
4. Optionally rejects outliers (e.g., from OS preemptions) using the interquartile range (IQR). If `reject_outliers=True`, drop any measured latency greater than `Q3 + 1.5 * IQR` before calculating the final p50 and p95. Q1 and Q3 are the 25th and 75th percentiles.

This will give us a much clearer picture of the true on-device performance.
