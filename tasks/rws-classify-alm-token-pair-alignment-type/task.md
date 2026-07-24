## Context

In aligned language model (ALM) knowledge distillation, a teacher model with its own subword vocabulary is distilled into a student model that may use a different vocabulary. Before transferring soft-label distributions, each student token must be mapped to one or more teacher tokens. The first step is to classify the textual overlap between a teacher token string $t$ and a student token string $s$.

The alignment function $\operatorname{align}(t, s)$ maps each pair to one of three categories:

$$\operatorname{align}(t, s) = \begin{cases} \texttt{exact} & \text{if } t = s, \\[4pt] \texttt{substring} & \text{if } (t \sqsubset s) \lor (s \sqsubset t), \\[4pt] \texttt{none} & \text{otherwise}, \end{cases}$$

where $t \sqsubset s$ denotes "$t$ is a proper substring of $s$" — that is, $t$ appears as a contiguous sequence of characters inside $s$ and $|t| < |s|$. The first branch takes precedence: when $t = s$ the label is always `"exact"`, even though the strings are trivially substrings of each other.

The substring check is essential in practice because BPE and SentencePiece vocabularies at different compression levels often produce tokens that overlap textually. For example, a student trained with fewer merge operations may emit the token `"hel"` where the teacher has `"hello"`.

A production ALM alignment pipeline processes $N$ such pairs, one per student token, and uses these labels to decide whether a direct 1-to-1 soft-label copy is valid (exact), whether a fractional split is needed (substring), or whether the student token has no teacher counterpart at all (none).

## Task

Implement `classify_alignment(pairs)`:

```python
def classify_alignment(pairs: list[tuple[str, str]]) -> list[str]:
    ...
```

Given a list of `(teacher_token, student_token)` string pairs, return a list of alignment labels — one of `"exact"`, `"substring"`, or `"none"` — for each pair.

Rules (applied in order):

1. **Exact match.** If $t = s$, label `"exact"`.
2. **Substring overlap.** Otherwise, if one string is a contiguous substring of the other (Python's `in` operator on strings), label `"substring"`.
3. **No match.** Otherwise, label `"none"`.

Use built-in Python string operations only — no external libraries required.

## Example

```python
pairs = [
    ("hello", "hello"),   # exact:  t == s
    ("hel",  "hello"),    # substring: "hel" in "hello"
    ("hello", "hel"),     # substring: "hel" in "hello"  (reversed direction)
    ("hello", "world"),   # none
]
classify_alignment(pairs)
# ["exact", "substring", "substring", "none"]
```

## What the gate checks

The gate computes the reference label for every test pair using the oracle algorithm — Python `==` and `in` on raw strings — and measures classification accuracy:

$$\texttt{exact\_match} = \frac{\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]}{N}$$

where $\hat{y}_i$ is the learner's label and $y_i$ is the oracle label for pair $i$. The gate requires $\texttt{exact\_match} = 1.0$ — every pair must be classified correctly.

Test data covers:

- Identical tokens of varying length.
- Teacher-substring-of-student pairs (short BPE fragment in longer student token).
- Student-substring-of-teacher pairs (reversed direction).
- Completely unrelated tokens (no character overlap).
- Edge cases: empty strings, single characters, multi-character substrings, and case-sensitive mismatches (`"NLP"` vs `"nlp"`).
