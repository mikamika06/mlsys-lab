Our iOS build fails on CI because the app bundle exceeds the 150MB App Store over-the-air download limit. The compiled binaries take 90MB, and we just bumped the LLM edge model from 60MB to 180MB for better context understanding, blowing way past the cap. The release team says we can only afford 40MB for the model payload inside the bundle.

I checked the PyTorch export, and it's literally just pumping out massive float32 tensors. We need to squish them down without destroying the user experience. The evaluation suite shows the current model gets 85.0 on our benchmark; product says we can sacrifice at most 1 point of accuracy (must remain >= 84.0).

Please build an export pass that measures exactly how many bytes each tensor costs, and provides routines for palettization, 8-bit quantization, and threshold-based sparsification. Then, use them to pack our model dictionary down below the 40MB ceiling.

Finally, we want a utility to compute the Pareto frontier of size vs accuracy from our sweep logs so we don't guess blindly next time. Ensure you add robust regression tests for it.
