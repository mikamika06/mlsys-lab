# Repetition report and num_predict budget

In the local runner's logs, complaints come in pairs. First: generation cuts
off tagged "stop: repetition", and when a support engineer looks at the raw
response token ids, there's no obvious repeat — they end up manually counting
tokens in the dump to figure out whether the trigger was fair. Second:
someone set `num_predict=-1` ("don't stop until the model itself says
enough") and got a three-token reply, while someone else set
`num_predict=-2` ("fill the context") and got a reply that ate up hundreds of
times more tokens than the model's context holds. Looks like somewhere in
the runner these two flags got computed with the same formula.

The repetition report needs to explain itself with numbers from the token
histogram, and the generation budget needs to be computed correctly for
every `num_predict` variant, including -1 and -2.

## What you write

`rundiag/histogram.py` — `build_histogram(tokens) -> dict[int, int]`,
frequency of each token id in the sequence.

`rundiag/report.py` — `repetition_report(tokens, window, threshold) -> dict`.
Look only at the tail — the last `window` tokens (or all of them, if there
are fewer). If some token id occurs `>= threshold` times in that tail, it's
a trigger; among the tokens with the highest count in the tail, take the one
with the smallest id (determinism on ties). Return:

```python
{
  "triggered": bool,
  "token": int | None,
  "window_count": int,
  "positions": [...],   # all indices of token in the FULL sequence, sorted; [] if not triggered
  "histogram": {...},   # histogram over the full sequence, token -> count
  "total_tokens": int,
  "unique_tokens": int,
}
```

`rundiag/predict.py` — `num_predict_budget(num_predict, prompt_tokens, context_size, hard_cap) -> int`.
`remaining = max(context_size - prompt_tokens, 0)`.

- `num_predict >= 0` — explicit limit, but no more than what's left of the
  context: `min(num_predict, remaining)`.
- `num_predict == -2` (fill the context) — exactly `remaining`, not a token
  more.
- `num_predict == -1` (infinite generation) — NOT bounded by context: the
  only limit is `hard_cap`, a separate and usually much higher ceiling:
  `max(hard_cap - prompt_tokens, 0)`.

`-1` and `-2` give different numbers whenever `hard_cap != context_size`.
Computing them with a single code branch is exactly the bug that broke the
users above.

## How it's graded

The grader computes the reference itself, from the same inputs: a set of
token sequences for the repetition report, a set of `(num_predict,
prompt_tokens, context_size, hard_cap)` for the budget. The third milestone
is yours: you write a test, and we swap `num_predict_budget` for a version
that collapses `-1` into the same branch as `-2`. Your test has to catch it.

```
mlsys project start m-explain-a-repetition-report-by-inspecting-the-emitte
mlsys project grade m-explain-a-repetition-report-by-inspecting-the-emitte --milestone 1
```
