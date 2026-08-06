# Ticket: High Perplexity Spike After Naive Layer Dropping in Small Transformer Depth Pruning

## Symptom

When deploying our compact transformer models for edge deployment, we apply depth pruning (layer dropping inspired by ShortGPT and Block Influence metrics) to reduce model size and latency. However, our current automated pruning pipeline causes an unacceptable degradation in validation perplexity. Specifically, when we drop layers from the model, the resulting language modeling loss spikes dramatically compared to the unpruned baseline, and random layer removal sometimes surprisingly outperforms our heuristic selection.

Engineers report that the computed Block Influence (BI) scores across transformer blocks appear misaligned with actual inter-layer dependencies, leading to critical layers being pruned while redundant layers are preserved. Furthermore, our layer selection logic does not properly account for boundary constraints or block ordering, and our perplexity evaluation script fails to accurately compare low-BI pruned models against random drop baselines under identical sequence contexts. We need to implement a robust Block Influence metric calculation, precise layer selection based on BI fixtures, and a rigorous perplexity evaluation harness to ensure depth-pruned models retain high linguistic fidelity without unexpected performance regressions.
