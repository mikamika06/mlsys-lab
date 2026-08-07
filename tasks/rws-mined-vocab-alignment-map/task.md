## Context

Cross-tokenizer knowledge distillation (distilling a teacher model into
a student with a *different* vocabulary/tokenizer) needs a way to map
each teacher token onto a student token before their logits can be
compared. **MinED** does this the simple, robust way: for every teacher
token string, find the student token string that is closest to it in
**edit distance** (Levenshtein distance — the minimum number of single-
character insertions, deletions, or substitutions to turn one string
into the other). No embeddings, no training — just string similarity,
which works because subword tokenizers built from similar corpora tend
to produce overlapping or near-identical pieces even when their merge
rules differ.

### Levenshtein distance

For strings $a$ (length $n$) and $b$ (length $m$), the edit distance is
the standard DP recurrence:
$$
D_{i,0} = i, \qquad D_{0,j} = j,
$$
$$
D_{i,j} = \min\Big(D_{i-1,j}+1,\ \ D_{i,j-1}+1,\ \ D_{i-1,j-1} + [a_i \ne b_j]\Big),
$$
and the distance is $D_{n,m}$.

## Task

Implement:

```python
def mined_vocab_align(teacher_vocab: list[str], student_vocab: list[str]) -> list[int]:
    ...
```

* `teacher_vocab` — list of teacher token strings.
* `student_vocab` — list of student token strings.

For each teacher token (in order), return the index into
`student_vocab` of the student token with the **minimum** edit distance
to it. On a tie, return the **smallest** such index (the first student
token achieving the minimum).

## Example

```python
teacher = ["running", "cat"]
student = ["run", "running", "cats", "dog"]
mined_vocab_align(teacher, student)
# "running" has edit distance 0 to student[1]="running" -> 1
# "cat" has edit distance 1 to student[2]="cats" ("cat"->"cats", 1 insertion),
#   and larger distance to everything else -> 2
# => [1, 2]
```

## What the gate checks

* **exact_match** — your returned index list must equal, element for
  element, a Python/DP oracle computing exact Levenshtein distances and
  taking the first argmin per teacher token, on several random
  teacher/student vocab pairs (fixed seed) — each case includes at least
  one exact-duplicate token to exercise the tie-break rule.
