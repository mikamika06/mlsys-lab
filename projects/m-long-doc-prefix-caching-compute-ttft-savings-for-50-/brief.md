We are seeing unacceptable Time-To-First-Token (TTFT) degradation in production for our new "Chat with your Library" feature. Users are uploading massive documents—often crossing 100,000 tokens—and then proceeding to ask a rapid series of 50 or more short analytical questions against that context within a single session.

Metrics show that for every single question submitted by the user, the GPU cluster is locking up while it grinds through a complete prefill sequence. It is re-ingesting the exact same 100,000-token document from scratch alongside the small new question string, resulting in quadratic attention costs that are spiking our compute footprint and driving latency into the tens of seconds per interaction.

I am told that advanced serving engines implement a feature called prefix caching, which allows the system to compute the Key-Value (KV) cache for the massive document exactly once, retain it in memory, and then only pay the computational cost for the small question suffix on subsequent turns. Before we overhaul our entire backend to support this, I need concrete numbers.

Please build a simulator to calculate the baseline prefill cost versus the cached prefill cost for this workload so we can justify the engineering investment.
