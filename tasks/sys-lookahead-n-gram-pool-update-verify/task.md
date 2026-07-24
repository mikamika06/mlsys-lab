## Context

Speculative decoding methods separate proposing future tokens from verifying them. A simple n-gram pool can act as the proposal mechanism: previously observed contexts provide likely next tokens, while the verifier accepts only proposals that match the actual continuation.

For a token sequence $x = (x_0, x_1, \dots, x_{m-1})$, an $n$-gram entry maps a context of length $n-1$ to possible next tokens:

$$
(x_i, \dots, x_{i+n-2}) \rightarrow x_{i+n-1}.
$$

The pool is updated as new verified tokens are observed. During lookahead, the current context proposes several future tokens. Verification compares the proposal with the trace, accepting the longest matching prefix. The accepted tokens are then inserted into the pool, similar to a Jacobi-style update where several positions are proposed before one verification pass.

## Task

Implement:

```python
def lookahead_pool_update_verify(
    trace: list[int],
    n: int,
    lookahead: int,
    pool_size: int
) -> tuple[list[int], list[tuple[tuple[int, ...], int]]]:
    ...
```

The function receives an integer token trace. It must simulate lookahead n-gram prediction and verification.

Rules:

1. Build an n-gram pool while scanning the trace. A context is the previous $n-1$ tokens and the value is the following token.
2. At each position where a prediction can be made, use the current pool to propose up to `lookahead` tokens. For each step, choose the most frequent next token for the current context. Break ties by choosing the smaller token value.
3. Verify the proposal against the remaining trace tokens. Append the verified matching tokens to the returned continuation list.
4. After verification, update the pool with all newly observed n-grams.
5. The returned pool must contain at most `pool_size` entries. Keep entries with larger frequency first, breaking frequency ties by context tuple and then token value.
6. Return the verified continuation list and the final ordered pool entries as a list of `(context, token)` pairs.

Use only the Python standard library.

## Example

```python
out, pool = lookahead_pool_update_verify(
    [1, 2, 3, 1, 2, 4],
    n=3,
    lookahead=2,
    pool_size=4,
)

# out contains tokens accepted by the verifier.
# pool contains ordered context -> next-token entries.
```

## What the gate checks

The gate runs the implementation on several token traces and compares both returned values with an independent reference implementation of the same n-gram pool algorithm.

The `exact_match` score must equal $1.0$. A solution that skips verification, updates the pool with unverified proposals, or uses a different ordering will fail.
