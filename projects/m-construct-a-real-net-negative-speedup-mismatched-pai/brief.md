We recently rolled out speculative decoding to production, combining a lightweight n-gram prompt lookup drafter with our 7B parameter target model. While initial benchmarks on standard instruction-following tasks showed an impressive 2.5x speedup, production telemetry is throwing severe alerts.

On certain customer prompts—especially long log files, code snippets, and repetitive data formats—throughput completely collapses. We are seeing wall-clock times for these requests running 30-40% slower than if we had just disabled speculative decoding entirely.

Furthermore, we are observing isolated cases where the n-gram drafter burns massive amounts of CPU cycles spitting out identical repeating sequences, which are then uniformly rejected by the target model. This drives the acceptance rate to effectively zero while maximizing draft overhead. Finally, the acceptance rate, typically ~70% on our internal eval sets, drops precipitously to ~15% when users submit prompts with out-of-domain vocabulary.

We need you to formalize these failure modes into a test suite. Please mathematically model the net-negative speedup, reproduce the degenerate n-gram loops, measure the acceptance collapse, and implement safety-net regression tests before we scale this system further.
