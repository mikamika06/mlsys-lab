## Context

The **Levenshtein edit distance** measures the minimum number of single-character
operations—insertions, deletions, and substitutions—required to transform one
string into another. It is a foundational primitive in spell-checking, DNA
sequence alignment, fuzzy search, and minimum-edit-distance (MinED) decoding
of token sequences in speech and translation pipelines.

Given a source string $s$ of length $m$ and a target string $t$ of length $n$,
the distance $D(s,t)$ is defined by the **Wagner–Fischer** recurrence. Let
$D[i][j]$ denote the edit distance between the prefix $s[0..i)$ and $t[0..j)$
(1-indexed for clarity). The boundary conditions are

$$D[0][j] = j \quad (0 \le j \le n), \qquad D[i][0] = i \quad (0 \le i \le m).$$

For $i \ge 1$ and $j \ge 1$ the recurrence is:

$$D[i][j] = \begin{cases}
D[i-1][j-1] & \text{if } s_i = t_j, \\[4pt]
1 + \min\!\Bigl(D[i-1][j],\; D[i][j-1],\; D[i-1][j-1]\Bigr) & \text{otherwise},
\end{cases}$$

where the three terms in the minimum correspond respectively to a **deletion**
(from $s$), an **insertion** (into $s$), and a **substitution**. The final
answer is $D[m][n]$.

A naïve recursive implementation explores $O(3^{\min(m,n)})$ branches. The
dynamic-programming table fills an $(m+1) \times (n+1)$ grid, giving
$O(mn)$ time and space. Space can be reduced to $O(\min(m, n))$ by keeping
only two rows (or one row with a rolling variable for the diagonal), but for
this task the full table or any $O(mn)$ variant is acceptable.

## Task

Implement the function:

```python
def edit_distance(source: str, target: str) -> int:
    """Return the Levenshtein edit distance between source and target."""
    ...
```

The function receives two plain Python strings and must return a single non-negative
integer. Any correct $O(mn)$ dynamic-programming formulation is accepted—full
matrix, two-row, or single-row with a stored diagonal. Do **not** use
`textwrap`, `difflib`, or any external library; implement the DP from scratch.

## Example

```python
edit_distance("kitten", "sitting")
# 3

edit_distance("saturday", "sunday")
# 3

edit_distance("", "")
# 0
```

For `"kitten"` → `"sitting"` the optimal alignment (cost 3) is:
substitute `k`→`s`, substitute `e`→`i`, insert `g`.

## What the gate checks

A single gate **`exact_match`**. For each of several test pairs—including empty
strings, identical strings, and strings up to length 200—the checker computes
the reference answer with its own canonical DP implementation and compares it to
your result. The gate equals `1.0` only when every distance is exactly correct;
any single mismatch yields `0.0`.
