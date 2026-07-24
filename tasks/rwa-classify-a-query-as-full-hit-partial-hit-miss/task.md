## Context

In many caching systems a new request is represented by a *query string*.  
The cache stores full strings, but the system only cares whether the query
matches the beginning of any cached entry.  
We distinguish three cases:

* **full‑hit** – the query is completely covered by some cached string,
  i.e. it is a prefix of that string (including equality).
* **partial‑hit** – the query shares a non‑empty common prefix with at least
  one cached string, but no cached string fully covers it.
* **miss** – the query has no common prefix with any cached string.

The longest common prefix length between two strings $a$ and $b$ is

$$\operatorname{lcp}(a,b)=\max \bigl\{k\mid a_{:k}=b_{:k}\bigr\}.$$

For a query $q$ and a cache $\mathcal C$, let

$$L_{\max}(q,\mathcal C)=\max_{c\in\mathcal C}\operatorname{lcp}(q,c).$$

The classification rule is then

$$
\text{classify}(q,\mathcal C)=
\begin{cases}
\texttt{full}   & \text{if } L_{\max}=|q|,\\[4pt]
\texttt{partial}& \text{if } 0<L_{\max}<|q|,\\[4pt]
\texttt{miss}   & \text{otherwise.}
\end{cases}
$$

## Task

Implement the function

```python
def classify_query(query: str, cache: list[str]) -> str:
    ...
```

It receives a query string and a list of cached strings and returns one of
the literals `"full"`, `"partial"` or `"miss"` according to the rule above.
The implementation must run in $O(|q|+|\mathcal C|)$ time per call.

## Example

```python
>>> classify_query("abc", ["abcd", "abx"])
'full'
>>> classify_query("abcde", ["abxyz", "a"])
'partial'
>>> classify_query("hello", ["world", "test"])
'miss'
```

## What the gate checks

The grader computes the reference classification for a set of test cases
using the exact algorithm described above.  
Your implementation must return exactly the same string for every case.
If any mismatch occurs, the `exact_match` metric is 0; otherwise it is 1.
