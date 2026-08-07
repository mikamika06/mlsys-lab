# Prefix Cache Cross-Tenant Isolation Vulnerability

## Symptom
Our multi-tenant LLM serving platform utilizes automatic prefix caching to reduce Time-To-First-Token (TTFT) and decrease prompt processing overhead. Recently, security auditing flagged that unauthorized request context might leak across tenants sharing the same model deployment.

Specifically, an attacker could mount a timing side-channel attack by sending carefully crafted prompts and measuring TTFT. Variations in prompt processing time allow an external adversary to infer whether specific prefix sequences (such as private system prompts, confidential documents, or system instructions) are currently cached in memory from another tenant's session.

## Problem Statement
The current prefix cache key generation scheme relies solely on prompt token content without incorporating cryptographic tenant boundaries or salt isolation. Consequently, identical token sequences submitted by different tenant IDs resolve to identical cache keys, sharing underlying KV cache physical blocks. Furthermore, the serving scheduler lacks isolation logic to enforce which request pairs may legally share cache blocks.

To eliminate this security boundary violation and side-channel vulnerability:
1. Implement a salted, tenant-partitioned cache key builder that computes secure hash keys incorporating tenant identifiers and unique salt state.
2. Build a timing oracle analyzer that detects potential prefix residency leaks by inspecting TTFT samples and evaluating request pair block-sharing safety under tenant isolation rules.
3. Establish safety verification tests to guarantee that tenant salt/key isolation cannot be bypassed or coalesced accidentally.
