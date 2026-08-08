We are deploying speculative decoding to speed up a 7B parameter target model. To maximize tokens per second, we want to apply INT8 weight-only quantization to our draft models, reducing their memory bandwidth requirements by half. However, we are seeing an unexpected regression: when using our tiny 100M parameter draft model, quantization actually *slowed down* our end-to-end token generation throughput.

Our profiling shows two things. First, the INT8 dequantization kernel has a slightly higher fixed launch overhead than the standard FP16 kernel. For very small models, this fixed overhead dominates the time saved from reading fewer bytes. Second, INT8 quantization slightly degrades the draft model's predictive accuracy, meaning our acceptance rate ($\alpha$) drops. Because a rejected token wastes the generation time for that token and subsequent ones, even a tiny drop in acceptance rate can severely punish throughput.

We need a tool to calculate the exact threshold where quantization becomes worth it.

The expected tokens generated per step in speculative decoding is the sum of $\alpha^i$ for $i$ from 0 to $K$. The time taken per step is $K \times T_{draft\_gen} + T_{target\_verify}$. The overall throughput is Expected Tokens divided by Step Time.

Implement a calculator that evaluates a list of draft model sizes. For each size, compute the expected throughput using FP16 and INT8 parameters. It must return the smallest draft size where the INT8 throughput strictly exceeds the FP16 throughput. If no such size exists, return `None`.
