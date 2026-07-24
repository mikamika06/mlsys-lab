## Context

Constrained ("grammar-guided") decoding forces an LLM to only ever emit
tokens that keep the output syntactically valid — e.g. a JSON tool-call
schema. At every decoding step the constraint engine must compute the set
of tokens the grammar allows next, given everything emitted so far, and
mask the logits of everything else to $-\infty$. For a context-free grammar
like JSON, computing that set is exactly running a **pushdown automaton**
(PDA): a stack of open-bracket obligations, where each transition either
pushes a new obligation (`{`, `[`), pops a satisfied one (`}`, `]`), or
advances the state at the top of the stack.

Consider this token-level JSON subset (values are treated as atomic token
*classes* — `STR`, `NUM`, `TRUE`, `FALSE`, `NULL` — not individual
characters):

$$
\begin{aligned}
\text{value}  &::= \texttt{STR} \mid \texttt{NUM} \mid \texttt{TRUE} \mid \texttt{FALSE} \mid \texttt{NULL} \mid \text{object} \mid \text{array} \\
\text{object} &::= \texttt{\{} \; \texttt{\}} \;\mid\; \texttt{\{} \; \texttt{STR} \; \texttt{:} \; \text{value} \; \big(\texttt{,} \; \texttt{STR} \; \texttt{:} \; \text{value}\big)^* \; \texttt{\}} \\
\text{array}  &::= \texttt{[} \; \texttt{]} \;\mid\; \texttt{[} \; \text{value} \; \big(\texttt{,} \; \text{value}\big)^* \; \texttt{]}
\end{aligned}
$$

Given a prefix of tokens already emitted (assumed to be a syntactically
valid partial derivation of this grammar), the PDA's stack — one frame per
currently-open `{...}` or `[...]`, each frame remembering whether it is
waiting for a key, a colon, a value, or a comma/close — fully determines
which tokens can legally come next.

## Task

Implement `allowed_next_tokens`:

```python
def allowed_next_tokens(prefix: list) -> list:
    ...
```

- `prefix` — a list of token-type strings from
  `{"{", "}", "[", "]", ":", ",", "STR", "NUM", "TRUE", "FALSE", "NULL"}`,
  a valid partial derivation of the grammar above (possibly empty).

Return a list (or set) of the token-type strings that could legally appear
immediately after `prefix`. If `prefix` is already a syntactically complete
top-level value, return an empty list (nothing more is allowed).

## Example

```python
allowed_next_tokens([])
# -> ['FALSE','NULL','NUM','STR','TRUE','[','{']   (start of any value)

allowed_next_tokens(['{'])
# -> ['STR', '}']            (first key, or an empty object)

allowed_next_tokens(['{', 'STR'])
# -> [':']                   (the key must be followed by a colon)

allowed_next_tokens(['{', 'STR', ':', 'NUM'])
# -> [',', '}']              (another member, or close the object)

allowed_next_tokens(['{', 'STR', ':', 'NUM', '}'])
# -> []                      (a complete top-level value; nothing follows)
```

## What the gate checks

The fixture holds every prefix (including the empty and the full-length
one) of 40 randomly generated, syntactically valid documents from this
grammar, at nesting depths up to 3 and mixing objects and arrays. For each
prefix the grader independently runs the same bracket-stack automaton
described above and compares the resulting allowed-token set to yours,
treating both as **sets** (order does not matter). Every one of the ~150
prefixes must match exactly (`exact_match == 1.0`). Forgetting that `STR`
plays two different roles (object key vs. value), mishandling the
empty-object/array shortcut, or not tracking nesting correctly will produce
a wrong set on some prefix.
