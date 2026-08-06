# Incident Report: Unintended Cache Hits Across Tenant Boundaries

## Symptom
During a recent multi-tenant deployment rollout of our prefix-cached KV serving infrastructure, security monitoring flagged potential cross-tenant memory contamination. Specifically, prompt prefix tokens submitted by Tenant A appear to be serving KV cache block hits for requests submitted by Tenant B. 

While prefix caching increases serving efficiency by reusing pre-computed KV states for shared system prompts, tenants must remain strictly isolated. Requests from different tenant IDs sharing identical token sequences should either maintain isolated block references or trigger explicit policy violations if cross-tenant block sharing is forbidden.

## Requirement
You need to build a log auditing tool that parses request traces from the serving engine, tracks block allocation and prefix tree hits per tenant, and identifies every instance where a request hits a KV block originally created by or assigned to a different tenant.

Your audit module must:
1. Reconstruct the state of the prefix cache from sequential execution logs (recording block allocations, token sequences, tenant IDs, and block lookup hits).
2. Detect illegal cross-tenant cache reuses where a request from tenant $T_2$ reuses a cache block containing state computed during a request from tenant $T_1$ ($T_1 \neq T_2$).
3. Compute exact leakage stats (number of illegally reused blocks, total leaked prompt tokens, and tenant violation pairs).
4. Implement a regression test suite in `tests/test_regression.py` that verifies the detector correctly flags unauthorized cross-tenant block sharing while allowing valid intra-tenant reuses.
