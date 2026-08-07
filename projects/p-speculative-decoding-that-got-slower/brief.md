We turned on speculative decoding in production after a successful demo. In the demo, latency dropped by 2x. In production, our p95 latency actually spiked, and overall throughput fell.

The demo was mostly single-batch queries with highly predictable outputs (code snippets). Production sees highly variable batch sizes and a mix of tasks, some of which are very hard to draft (low acceptance rate). Speculative decoding drafts $\gamma$ tokens, taking $\gamma \times t_{draft}$ time, and then verifies them in one large model pass taking $t_{verify}$. If the model rejects most of the draft, we just paid for the draft and the heavy verification but got only 1 token out of it, which is slower than just running the standard generation step $t_{model}$.

We need to understand exactly when speculative decoding is a net positive. Your task:
1. Measure the real acceptance rate from traces.
2. Build an analytical model of the speedup based on $\gamma$ and the acceptance rate $\alpha$.
3. Extend the model to account for batch size $B$, since larger batches make $t_{verify}$ more expensive and dilute the benefits.
4. Compute the threshold acceptance rate at which speculation becomes a net loss.
5. Ensure our p95 latency does not degrade by selectively disabling speculation when it's predicted to be slower than the baseline.
6. Implement an adaptive policy that tracks moving average $\alpha$ per request and turns off speculation dynamically to prove a consistent win.
