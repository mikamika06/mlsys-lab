## Context

A continuous-batching inference server packs each scheduling step against a
fixed **token budget** $B$ — the maximum number of tokens (decode + prefill)
it can afford to run through the model this step. Two kinds of work compete
for that budget:

- **Decode**: every already-running sequence needs exactly **one** token
  this step to keep making progress. With $R$ running sequences that costs
  $R$ tokens.
- **Chunked prefill**: a pending request still owes $P$ prefill tokens; the
  scheduler may spend any leftover budget processing a chunk of it.

A correct packer **reserves decode first**:

$$
d = \min(R, B), \qquad p = \min\big(P,\ B - d\big),
$$

so every running sequence gets its token before anything is spent on
prefill. If it instead spent the budget on prefill first, a chunk large
enough to exhaust $B$ would leave $B - p < R$ tokens for decode — some
already-running sequences get **zero** tokens that step. Each dropped
sequence stalls for a full step, and the per-token latency (ITL) for every
affected request spikes.

## Task

Fix `pack_step`:

```python
def pack_step(token_budget: int, num_running: int, prefill_remaining: int) -> tuple:
    ...
```

- `token_budget` — $B$, tokens available this step.
- `num_running` — $R$, number of already-running sequences (each needs one
  decode token).
- `prefill_remaining` — $P$, prefill tokens still owed by the pending
  chunked-prefill request (`0` if there is none).

Return `(decode_tokens, prefill_chunk)` following the reserve-decode-first
rule above. The current implementation spends the budget on the prefill
chunk first and gives decode whatever is left — find and fix the bug.

## Example

$B=16$, $R=4$, $P=64$ (a large pending prefill and a full budget of running
sequences):

- **Correct**: $d = \min(4,16) = 4$, $p = \min(64, 16-4) = 12$ → `(4, 12)`.
  All 4 running sequences get their decode token.
- **Buggy** (prefill first): $p = \min(64,16)=16$, $d=\min(4,16-16)=0$ →
  `(0, 16)`. All 4 running sequences are starved this step.

## What the gate checks

The grader replays a fixture of `(token_budget, num_running,
prefill_remaining)` snapshots recorded from a scheduler run, including
several deliberately budget-starved ones, and computes the reference
`(decode_tokens, prefill_chunk)` for each with the reserve-decode-first
formula. Your output must **exactly** match the reference on every
snapshot (`exact_match == 1.0`); an implementation that starves decode
whenever the prefill chunk alone could exhaust the budget will disagree on
the starved cases while still matching on the easy ones.
