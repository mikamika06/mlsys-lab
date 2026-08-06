We have received a critical bug report from the model quantization team. They are implementing the GPTQ algorithm to quantize a new suite of 7B and 13B models. However, they are experiencing significant perplexity degradation, to the point where the models output gibberish after the process.

The team has identified three specific symptoms:
1. On certain transformer blocks, the quantization script crashes with a singular matrix error when attempting to compute the Cholesky decomposition of the Hessian. They need an adaptive mechanism to find the minimum damping factor that allows the Cholesky decomposition to succeed.
2. The activation ordering (`act_order=True`) seems to be prioritizing the least important weights instead of the most important ones. The algorithm is supposed to process columns sorted by their total activation variance (which corresponds to the diagonal entries of the Hessian).
3. The newly implemented "lazy batching" update mechanism—designed to speed up the application of quantization errors—is drifting. The output of the blocked update does not perfectly match the output of a naive step-by-step sequential update.

Please implement the core algorithmic mechanics in `gptq/core.py`. You will need to calculate the correct permutation vector, adaptively discover the required Cholesky damping, and perfectly apply the lazy-batch weight updates. Finally, write a safeguard regression test that strictly catches if the permutation vector sorts weights in the wrong direction.
