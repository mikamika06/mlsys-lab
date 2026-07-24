## Context

In a multi‑tenant cache, each request is tagged with a *cache salt* that isolates its cached blocks from other tenants.  
The block chain hash for the first block (block 0) is computed by hashing together the request token and the salt:

$$h_0 = \operatorname{SHA256}\bigl(\texttt{token} \,\|\, \texttt{salt}\bigr).$$

Two requests are said to *collide* if their $h_0$ values are identical; otherwise they are *isolated*.  The hash must be deterministic and independent of the process, so we use a standard cryptographic hash function.

## Task

Implement `blocks_collide(req1, req2)`:

```python
def blocks_collide(req1: dict, req2: dict) -> bool:
    ...
```

Each request is a dictionary with keys:

* `token` – a string identifying the resource.
* `cache_salt` – an integer (can be negative).

The function must compute the block‑0 hash for each request using SHA256 on the UTF‑8 bytes of the token concatenated with the 4‑byte big‑endian representation of the salt.  
It should return `True` if the two hashes are equal, otherwise `False`.

## Example

```python
req_a = {"token": "abc", "cache_salt": 1}
req_b = {"token": "abc", "cache_salt": 2}

print(blocks_collide(req_a, req_b))   # False – isolated
```

If the salts were both `1`, the function would return `True`.

## What the gate checks

The grader recomputes the reference block‑0 hashes with the same algorithm and compares your output.  
All test pairs must match exactly (`exact_match == 1.0`).  No other metrics are evaluated.
