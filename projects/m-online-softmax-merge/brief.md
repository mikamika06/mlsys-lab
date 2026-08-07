# Ticket: Softmax Accumulation and Numerical Drift in Blocked Attention

## Symptom
During long-context FlashAttention development, downstreams reported subtle output drift when scaling sequence lengths past 4K tokens on FP32 and FP16 mock-up pipelines. While standard tile-by-tile attention produces clean outputs, our experimental tiled attention kernels drift significantly relative to PyTorch standard attention.

A diagnostic dump reveals that across varying sequence lengths, naive accumulation of intermediate tile outputs suffers from numerical precision degradation and scales poorly with sequence length. Furthermore, our internal validation checks lack standard tolerance contracts, making it impossible to establish an automated gating tolerance harness that catches online softmax merging regressions.

## Proposed Task
Implement online softmax merging arithmetic that numerically matches full-sequence attention across chunked computation tiles. Build an upstream-style tolerance harness that validates relative errors and verifies how numerical degradation scales as sequence length increases.
