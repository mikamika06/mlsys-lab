Ticket #8409: Edge model dispatch pipeline failing integrity and device capability checks.

Our nightly edge deployment pipeline failed during the v2.4 model release sync. The automated release validator flagged three separate failure conditions across target client platforms during staging:

First, several legacy mobile devices running older OS revisions received model packages requiring feature flags like metal3 and specialized fp16 dot-product acceleration. This led to instant initialization crashes for approximately 4% of active devices in the staged canary ring.

Second, binary hashes computed by the model conversion script on build worker A did not match the SHA-256 digests produced by build worker B for identical model weight structures. This key-ordering and float-formatting variance caused widespread CDN cache misses and asset verification failures.

Third, the variant manifest selector bundled an unconstrained set of full-precision and quantized variants, exceeding the strict 150MB total asset download budget for edge deployment bundles and blocking over-the-air package generation.

We need to formalize the edge export pipeline inside `edgeexport/`:
1. Calculate effective OS floors based on required hardware features and determine eligible vs. excluded devices with explicit rejection reasons in `edgeexport/filtering.py`.
2. Implement canonical, bit-reproducible manifest serialization and hashing in `edgeexport/convert.py`.
3. Implement budget-constrained variant set selection in `edgeexport/selection.py`.
4. Include regression testing in `tests/test_regression.py` to ensure budget bounds are enforced.
