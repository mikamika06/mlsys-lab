We have an urgent issue with our local vLLM serving cluster. Our new conversational agent works perfectly for the first few turns of a session, getting sub-second responses. But inevitably, right around turn 4 or 5, the time-to-first-token spikes massively.

We looked at the server metrics, and it turns out that despite passing `--enable-prefix-caching`, the prompt cache is completely missing for these later turns. Instead of reusing the previously processed context, the engine is forced to process 10,000+ tokens of system instructions and history from scratch!

The prompt we send to the server consists of three concatenated parts: a massive static system prompt, a small dynamic block (which contains things like the current timestamp and live status, which updates mid-conversation), and the growing conversation history.

We need you to analyze the block logs to identify exactly which turn breaks the prefix cache and compute how many blocks are being wasted. Once you pinpoint the flaw in how we stitch these blocks together, redesign the prompt layout to guarantee maximum prefix reuse. The dynamic status block must not poison the shared prefix of the long, static conversation history.
