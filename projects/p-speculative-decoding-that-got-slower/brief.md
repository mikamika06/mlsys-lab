# Ticket: Speculative Decoding Made Our Production Service Slower

## Symptom

During initial local benchmark demos and controlled offline experiments, enabling speculative decoding with our small draft model provided a promising $1.6\times$ to $2.1\times$ speedup in end-to-end token generation latency. However, after deploying the speculative decoding pipeline to our production LLM inference clusters last week, the average service throughput actually dropped by ~18%, and high-concurrency $P95$ request latencies degraded significantly.

Operations monitoring shows that under realistic production traffic—which features variable batch sizes, highly non-uniform prompt lengths, and shifting domain distributions—the draft model's acceptance rate fluctuates wildly. In high-concurrency regimes, the overhead of draft token generation and verification steps outweighs the savings from accepted tokens.

## Goal

We need to mathematically model and dynamically control speculative decoding in our serving system. You must write an adaptive speculative decoding scheduler and execution manager that:

1. Measures online acceptance rates across streaming request contexts.
2. Models the theoretical and real-world speedup bounds as a function of draft length $\gamma$, verification overhead $\alpha$, and acceptance rate $\tau$.
3. Accounts for batching dynamics, recognizing when large batch sizes make draft execution compute-bound rather than memory-bound.
4. Dynamically throttles or completely disables draft speculative steps when traffic conditions or acceptance drop below profitability thresholds.
5. Guarantees that $P95$ request latency under speculative decoding is strictly no worse than baseline non-speculative execution across all batch regimes.
6. Implements an adaptive policy with provable speedup gains across non-stationary traffic distributions.
