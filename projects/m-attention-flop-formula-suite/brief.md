Ticket FLOP-882: Inaccurate MFU and TFLOPs accounting in profiling pipeline

Our cluster telemetry pipeline is reporting inconsistent Model FLOPs Utilization (MFU) across training runs for FlashAttention and packed variable-length (varlen) sequences. Profiling logs show MFU dropping inexplicably when switching from fixed-length dense batches to concatenated varlen batches, and backward-pass TFLOPS calculations are wildly off when accounting for SwiGLU MLPs and Grouped-Query Attention (GQA) architectures.

Investigation shows that different sub-teams are using ad-hoc FLOP approximations in their accounting scripts: some treat causal attention as exactly half of non-causal attention without handling sequence length asymmetry, others ignore GQA key/value projection head count ratios, and backward pass multipliers are hardcoded as 2x total iterations instead of keeping forward (1x), backward (2x), and combined (3x) accounting distinct.

We need a unified `flopcount` package to accurately compute attention GEMMs for causal and non-causal GQA/MHA, calculate exact attention FLOPs for varlen sequence length histograms, and track forward, backward, and total forward+backward FLOPs across full Transformer layers and models.
