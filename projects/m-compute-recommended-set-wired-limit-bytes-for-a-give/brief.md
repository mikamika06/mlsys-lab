Our users running local LLMs on Apple Silicon are reporting system lockups and abrupt crashes (kernel panics) during long generation sessions. The models load fine initially, but as text generation continues for a few hundred tokens, the system becomes unresponsive, beachballs, and then the app gets SIGKILLed or the whole Mac reboots.

We monitored `top` during generation and noticed that our model loads cleanly into unified memory, but as tokens are generated, the RSS climbs linearly without bound until macOS forcefully terminates the process for exceeding the default wired memory limit.

Apple's default wired limit is very conservative (often ~66% of system RAM). We need to maximize our context window by safely raising `mx.metal.set_wired_limit()` up to 85% of total memory. However, we must leave at least 3GB for the OS on <=16GB systems, 6GB on <=32GB, and 8GB on larger ones.

Raising the wired limit isn't enough; we also need to clamp the cache size using `mx.metal.set_cache_limit()` so that `model_bytes + cache_limit` never exceeds the new wired limit (leaving a 256MB safety buffer for intermediate allocations).

Can you write a tuning utility that takes `hw.memsize` and the model size, computes the optimal wired and cache limits, and simulates the expected RSS growth over N tokens to prove the memory usage plateaus safely?
