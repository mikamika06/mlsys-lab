# Ticket: Post-Pruning Accuracy Recovery and Fine-Tuning Protocol

## Problem Description
Following our recent model compression pass, the edge inference classifier was pruned by 50% magnitude sparsity to meet strict memory limits. While model footprint was reduced as expected, validation accuracy suffered an immediate drop of approximately 6%.

Small-scale experiments suggest that a brief fine-tuning phase ("healing") can restore most of the lost accuracy. However, naive fine-tuning attempts cause gradient updates to populate zeroed weight slots, causing density to return ("sparsity leakage"). Furthermore, unconstrained fine-tuning risks overshooting step budgets or causing training instability.

## Requirements
We need an automated pruning and healing pipeline that:
1. Captures and records baseline accuracy and loss metrics immediately after magnitude pruning.
2. Configures and enforces a step budget for fine-tuning to bound compute overhead.
3. Zeroes out gradients on pruned weights during fine-tuning to prevent sparsity leakage.
4. Tracks the trajectory of loss reduction across fine-tuning steps.
5. Recovers at least 70% of the accuracy lost during pruning while staying strictly within step budget.
6. Includes regression tests that catch unmasked gradient updates and step budget overruns.
