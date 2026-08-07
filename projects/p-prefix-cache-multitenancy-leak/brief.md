We recently rolled out the prefix cache to production and it has been a massive success, significantly improving TTFT (Time To First Token) across the board. However, our security audit team just flagged a critical P0 incident: the prefix cache is leaking data across different tenants.

Specifically, they discovered a case where one client's response was accelerated because another client had recently submitted the exact same prefix in their prompt. While the LLM generation itself is mathematically safe, this overlapping state creates a timing side-channel. An attacker could brute-force prompt prefixes and observe generation times to verify if another tenant is querying specific sensitive strings.

We need strict multitenant isolation immediately. However, if we just blindly isolate everything by `tenant_id`, we will destroy our hit rate because virtually all tenants share the exact same system prompts.

Your task is to implement tenant isolation in the caching layer while maintaining high hit rates by safely sharing explicitly marked system prefixes across all users. Fix the data leak without losing our performance gains. Build it to pass all six validation milestones.
