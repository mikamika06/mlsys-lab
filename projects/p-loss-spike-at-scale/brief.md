# Ticket: Loss spikes at large scale

We are encountering a critical training stability issue when scaling our large model training job to 64 nodes, while the exact same model configuration runs smoothly and stably on 8 nodes. The problem manifests as sudden, intermittent loss spikes during training, occasionally leading to complete divergence of the model weights and training collapse.

We have conducted initial data integrity audits and confirmed that input data batches are distributed identically across both setups. Preliminary diagnostics suggest that the root cause stems from numerical precision effects and accumulation errors under massive distributed scaling, rather than dataset anomalies or hardware memory faults.

We need a systematic suite of diagnostics, simulation tools, and regression tests to isolate the source of numerical error, verify reduction determinism, locate the specific operation accumulating error, ensure convergence without spikes, and implement a robust detector to catch these anomalies early.
