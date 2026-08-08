We are running embedding generation endpoints locally and observing significant tail latency degradation and low GPU/CPU utilization when serving requests individually compared to batch processing.

### The Symptom
During peak load tests against our local embedding runner, we send 256 individual short text payloads sequentially or concurrently in small trickles, resulting in poor throughput and high request overhead. When we switch to sending all 256 items in a single batched request, throughput skyrockets by a factor dictated by our `throughput_ratio` gate constraint, yet unexpected downstream behavior occurs when clients try to compare embedding vectors. Specifically, cosine similarity scores between vectors collapse or become completely meaningless when clients inadvertently mix vectors from different embedding models, or when they assume raw model outputs are L2-normalized when they are actually unnormalized vectors.

### The Objective
To resolve these systemic efficiency and correctness issues in our local embedding runner pipeline, we need a complete Python module that:
1. Implements a performance comparison utility that calculates throughput and ratio metrics between executing 256 single embedding calls versus one batched embedding call using deterministic mock latency or tensor profiles without external deep learning frameworks (using pure Python and NumPy).
2. Implements a detector that reliably identifies whether an embedding model or output tensor returns L2-normalized vectors by checking if the vector norms equal 1.0 within a tight floating-point tolerance.
3. Implements an analysis helper that demonstrates and catches why mixing two different embedding models breaks cosine similarity comparisons, highlighting the vector space misalignment.
4. Provides a robust regression test suite in `tests/test_regression.py` that enforces these invariants, ensuring that future code modifications do not reintroduce model mixing errors or broken normalization assumptions.
