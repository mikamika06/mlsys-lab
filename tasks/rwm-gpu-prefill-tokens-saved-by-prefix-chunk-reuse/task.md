## Context

When a GPU prefill engine processes a sequence of tokens, it can avoid recomputing work for any prefix that has already been computed in an earlier request.  
If the current request shares a contiguous subsequence with a previously cached chunk, the engine may reuse that chunk and skip the corresponding tokens.  

Let a *trace* be an ordered list of requests
\[
T = (R_1,R_2,\dots ,R_m), \qquad R_k \subseteq \mathbb{Z}^+,
\]
where each request \(R_k\) is a finite sequence of token identifiers.  
For a given maximum chunk length \(L_{\max}\) we maintain a cache that contains **all** contiguous subsequences of every processed request whose length does not exceed \(L_{\max}\).  

When a new request arrives, the engine finds the longest prefix
\[
P_k = R_k[0:\ell]
\]
that is already present in the cache.  
The number \(\ell\) of tokens saved for that request equals the length of this longest cached prefix.  
After processing \(R_k\), all its contiguous subsequences up to length \(L_{\max}\) are inserted into the cache so they can be reused by later requests.

This problem asks you to implement a function that, given a trace and a maximum chunk size, returns the total number of tokens saved across the entire trace.

## Task

Implement `prefix_reuse_savings(trace, chunk_size=512)`:

```python
def prefix_reuse_savings(trace: list[list[int]], chunk_size: int=512) -> int:
    ...
```

* `trace` – a list of requests; each request is itself a list (or array) of integer token IDs.
* `chunk_size` – the maximum length of any cached subsequence.  
  The default value \(512\) matches typical GPU chunk sizes.

The function must return an **integer** equal to

\[
\sum_{k=1}^{m} \ell_k,
\]

where \(\ell_k\) is the length of the longest prefix of request \(R_k\) that appears in the cache built from all previous requests.  
The cache starts empty.

Your implementation must be fully deterministic and use only standard Python data structures (lists, tuples, sets).  No external libraries are required.

## Example

```python

trace = [
    [1, 2, 3],
    [2, 3, 4],
    [1, 2, 5]
]

saved = prefix_reuse_savings(trace, chunk_size=512)
print(saved)   # → 4
```

Explanation:

* Request 1: cache empty → no saved tokens.
* Request 2: longest cached prefix is `[2,3]` (length 2).
* Request 3: longest cached prefix is `[1,2]` (length 2).

Total saved = \(0 + 2 + 2 = 4\).

## What the gate checks

The grader computes a reference answer by running an exact algorithm that enumerates all contiguous subsequences up to `chunk_size`.  
Your implementation must produce exactly the same integer for every test case.  
If the returned value differs, the `exact_match` metric will be set to 0 and the task will fail.
