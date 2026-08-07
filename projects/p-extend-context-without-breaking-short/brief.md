Team, we have encountered a severe performance regression across our Mixture-of-Experts serving infrastructure following our recent integration of context extension techniques.

To support ultra-long user prompts, we recently applied RoPE scaling across our attention layers. While this successfully expanded our maximum context length and improved needle-in-a-haystack retrieval performance on very long sequences, it introduced an unexpected and critical degradation on standard short queries. Latencies for short prompts have increased, and output quality, coherence, and accuracy on regular-length queries have noticeably deteriorated.

We cannot afford to sacrifice short-prompt efficiency and accuracy for long-context gains. Both operational regimes must function at peak performance concurrently.

Your mission is to investigate, implement, and validate a robust context scaling strategy that successfully maintains support for long-context requests while strictly preserving or exceeding baseline accuracy and performance metrics on short inputs. You will build a comprehensive suite of milestones and regression tests to measure degradation, evaluate different scaling methods, tune hyperparameters for dual-regime performance, verify context retrieval accuracy, enforce zero short-prompt regression, and construct an automated testing suite that robustly catches both failure modes.
