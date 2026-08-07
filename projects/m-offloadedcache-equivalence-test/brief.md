Our inference backend is transitioning to the new `transformers` Cache API. As part of this migration, we've developed an `OffloadedCache` that aggressively moves inactive KV pairs to host memory to free up VRAM during long generation passes.

However, we are seeing alarming bug reports from production:
1. Legacy compatibility is breaking. Older models often crash because they expect `past_key_values` to be a plain tuple of tuples, rather than a stateful `Cache` object.
2. Silent corruption. There are suspicions that `OffloadedCache` might occasionally misalign tensor concatenations during offloading, causing outputs to subtly diverge from the standard `DynamicCache`.

Your task is to fix these gaps:
1. **Detect and port legacy usage**. Implement `is_legacy_tuple` and `port_legacy_to_cache` in `kvcache/legacy.py` to seamlessly convert a standard `Tuple[Tuple[np.ndarray]]` to a `DynamicCache` instance.
2. **Implement OffloadedCache** in `kvcache/offloaded.py`. It should adhere to the exact identical interface as `DynamicCache`, capturing and appending KV states correctly while conceptually storing them in a separate offloaded dictionary.
3. **Build the Equivalence Safeguard** in `tests/test_regression.py`. Your test must initialize both caches, feed them identical sequential token generation updates, and assert they yield mathematically identical key/value outputs. We will purposefully inject a truncating bug into your cache to ensure your test catches it.
