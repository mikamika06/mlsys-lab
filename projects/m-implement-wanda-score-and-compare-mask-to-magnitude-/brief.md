We are serving a growing number of large language models and running into severe memory bottlenecks. To alleviate this, we are building a one-shot post-training pruning pipeline. Right now, our pruning scripts are entirely naive—we just drop the weights with the smallest absolute values (magnitude pruning). This approach completely ignores the actual activations flowing through the model, leading to catastrophic perplexity degradation even at moderate sparsity levels.

We need you to implement two advanced one-shot pruning techniques to fix this:

First, implement **Wanda** (Weight Activation and Normalization), which scores weights by multiplying their magnitude by the L2 norm of the corresponding input feature activations. This simple change avoids pruning small weights that process critical, large-magnitude features.

Second, implement a simplified single-layer **SparseGPT** using Optimal Brain Surgeon (OBS) math. Instead of just masking weights, this method uses the inverse Hessian of the inputs to iteratively prune weights and simultaneously update the remaining weights to compensate for the loss.

Your tasks:
1. Implement `magnitude_mask`, `wanda_mask`, and `mask_recall` in `pruning/wanda.py`. For Wanda, the score for $W_{ij}$ is $|W_{ij}| \cdot ||X_j||_2$. Both methods should prune `k = int(in_features * sparsity)` weights per row.
2. Implement `obs_prune` in `pruning/sparsegpt.py`. Compute the Hessian $H = (X^T X)/N$, add dampening $\lambda = 0.01 \cdot \text{Tr}(H)/d$ to the diagonal, and invert it. For each row, iteratively (for $k$ steps) prune the weight with the lowest $W_{ij}^2 / H^{-1}_{jj}$ and update the remaining weights: $W_{i} \gets W_{i} - W_{ij} H^{-1}_{j,:} / H^{-1}_{jj}$.
3. Measure how well Wanda preserves output quality by evaluating the Mean Squared Error (MSE) curve of a pruned `TinyLM` over various sparsities in `pruning/eval.py`.
4. Write a regression test proving that your OBS implementation actually modifies the unpruned weights compared to just masking them.
