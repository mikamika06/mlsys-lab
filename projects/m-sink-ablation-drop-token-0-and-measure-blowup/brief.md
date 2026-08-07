We've been experimenting with aggressive eviction policies in our KV cache serving infrastructure to see how much memory we can claw back. One of the teams proposed a "drop early tokens" policy, starting by aggressively evicting the very first token (token 0) from the cache on the assumption that its information value decays immediately after the initial few steps of generation.

We deployed this to staging, and the symptom was immediate: the model started outputting complete gibberish (punctuation loops, repeating words) almost exactly when token 0 was evicted. We observed that the attention probabilities for the remaining tokens seemed to scale upwards dramatically right after the eviction, causing a cascading failure in the output distribution.

We need a diagnostic toolkit to investigate this blowup behavior offline.

First, implement `reconstruct_mask` to parse our JSON-like cache dumps and return a boolean mask of which tokens were actually kept per layer and head.

Second, build `drop_token_0` to simulate this eviction on a given probability matrix. It should force attention to key 0 to be 0.0 for all queries $q > 0$, and renormalize the remaining keys so they sum to 1.0.

Third, write `measure_blowup` to quantify the magnitude of this cascade, returning the maximum absolute difference between the original and ablated matrices.

Finally, we run these on live model states. Write a regression test verifying that your ablation simulation doesn't mutate the input matrices in-place, which would permanently poison the live model hook.
