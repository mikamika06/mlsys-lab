## Context

A continuous-batching inference server processes requests in discrete
iterations, each iteration bounded by a **token budget** $B$ — the max
number of tokens (prefill + decode, combined) it will pack into one
iteration. Two client-facing latencies come out of this:

- **TTFT** (time to first token): how long a request waits, from arrival,
  until its first generated token appears.
- **ITL** (inter-token latency): the gap between successive generated
  tokens of an already-streaming request. The *tail* ITL — the worst gap
  any request ever experiences — is what users actually notice as
  "stutter."

The budget is a genuine trade-off knob. A **larger** $B$ lets a big prompt
finish prefilling in fewer iterations (lower TTFT for that request) — but
each iteration that packs a large prefill chunk alongside ongoing decode
work takes proportionally longer in wall-clock time (iteration cost scales
with tokens processed), which stalls every other request's next token for
that whole iteration (worse tail ITL). A **smaller** $B$ keeps iterations
short (low tail ITL) at the cost of needing many more iterations — hence
more wall-clock time — to push a big prompt through prefill (worse TTFT).

## Task

Implement `choose_budget_min_tail_itl(workload, candidate_budgets, ttft_cap)`:

```python
def choose_budget_min_tail_itl(workload: list[dict], candidate_budgets: list[int], ttft_cap: int) -> int:
    ...
```

- `workload`: list of request dicts, each
  `{"id": int, "arrival": int, "prompt_len": int, "decode_len": int}`
  (`id` unique; `prompt_len >= 1`, `decode_len >= 1`).
- `candidate_budgets`: list of positive int budgets to evaluate.
- `ttft_cap`: int, the max allowed worst-case TTFT.

**Scheduler model**, simulated per candidate budget $b$ over discrete
iterations with a running wall-clock time $T$ starting at $0$:

1. Requests prefill strictly FCFS, ordered by `(arrival, id)`; only **one**
   request prefills at a time (its remaining prompt tokens are consumed
   over as many iterations as needed).
2. Each iteration, **every** currently-decoding request unconditionally
   receives exactly 1 token — say there are $k$ of them this iteration.
3. Whatever budget remains, $\max(b-k,\,0)$, is spent continuing the
   current prefill target: $\text{chunk} = \min(\text{leftover},\ \text{remaining\_prompt})$.
4. The iteration's duration is $k+\text{chunk}$ (tokens actually
   processed); $T \mathrel{+}= \text{duration}$, and every token produced
   this iteration is timestamped at the new $T$.
5. If the prefill target's prompt fully empties this iteration, it starts
   decoding from the **next** iteration onward.
6. If nothing can happen this iteration (no active decoders, no prefill
   target has arrived yet), fast-forward $T$ to the next arrival — no
   iteration is spent idling.

For each request: $\text{TTFT} = (\text{timestamp of its first token}) - \text{arrival}$;
$\text{ITL} = $ the largest gap between its consecutive token timestamps
(0 if it produces fewer than 2 tokens). Per budget, $\text{max\_ttft} =
\max_i \text{TTFT}_i$, $\text{max\_itl} = \max_i \text{ITL}_i$.

Return the candidate budget with the smallest `max_itl` among those with
`max_ttft <= ttft_cap` (ties broken by the smaller budget value). If
**no** candidate satisfies the cap, return `-1`.

## Example

```python
workload = [
    {"id": 0, "arrival": 0, "prompt_len": 30, "decode_len": 6},
    {"id": 1, "arrival": 1, "prompt_len": 5,  "decode_len": 8},
    {"id": 2, "arrival": 3, "prompt_len": 40, "decode_len": 6},
]
choose_budget_min_tail_itl(workload, [4, 8, 16, 32], ttft_cap=100)
# simulates each of 4/8/16/32; among those meeting the cap, returns
# whichever has the smallest worst-case inter-token gap
```

## What the gate checks

The oracle runs this exact simulation (pure integer arithmetic — no
floats, so results are exactly reproducible) on three fixed workload /
candidate-list / cap combinations: two where the cap excludes the
smallest candidate(s) on TTFT grounds (so "just return the smallest
budget" fails), and one where **no** candidate satisfies the cap (so the
`-1` sentinel path must be implemented too). Your returned integer is
checked for exact equality (`exact_match`) against the oracle's chosen
budget on every case — getting the iteration-duration formula, the
FCFS prefill-target selection, the fast-forward-on-idle rule, or the
tie-break wrong will pick a different (or infeasible) budget on at least
one case.
