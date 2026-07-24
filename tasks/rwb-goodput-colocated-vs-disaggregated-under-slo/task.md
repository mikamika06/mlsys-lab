## Context

Two ways to serve a batch of LLM requests on GPUs:

- **Colocated** — one engine runs both the prefill (prompt-processing) and
  decode (token-generation) phases of every request, interleaved via
  continuous batching. This is efficient, but a newly-arrived request's
  prefill chunk competes for the same compute as an in-flight request's
  decode step — every decode step that overlaps a concurrent prefill gets
  slower. This is the well-documented "prefill-interrupts-decode" problem
  that motivates disaggregation in systems like DistServe / Mooncake.
- **Disaggregated** — prefill and decode run on *separate* engine pools, so
  decode steps are never slowed down by a concurrent prefill. The cost is a
  KV-cache handoff between the two pools, adding a fixed latency to every
  request's time-to-first-token (TTFT).

A request is only useful if it meets BOTH of its latency SLOs:

$$
\text{TTFT}_i \le \text{ttft\_slo}_i \qquad \text{AND} \qquad \text{ITL}_i \le \text{itl\_slo}_i
$$

where ITL ("inter-token latency") is the time between consecutive generated
tokens. **Goodput** is the count of requests that meet both SLOs.

## Task

Implement `goodput_colocated_vs_disaggregated`:

```python
def goodput_colocated_vs_disaggregated(arrival, prompt_len, output_len, ttft_slo, itl_slo):
    ...
```

* `arrival` — 1-D float array, arrival time (seconds) of each request.
* `prompt_len`, `output_len` — 1-D float/int arrays, prefill and decode
  token counts.
* `ttft_slo`, `itl_slo` — 1-D float arrays, per-request SLO thresholds
  (seconds).

Using the fixed constants `PREFILL_THROUGHPUT` (tokens/sec), `DECODE_RATE`
(tokens/sec/request), `INTERFERENCE_FACTOR`, and `TRANSFER_LATENCY`
(seconds) given in the starter, compute for every request $i$:

$$
\text{prefill\_dur}_i = \frac{\text{prompt\_len}_i}{\text{PREFILL\_THROUGHPUT}}, \qquad
\text{ttft\_time}_i = \text{arrival}_i + \text{prefill\_dur}_i, \qquad
\text{decode\_end}_i = \text{ttft\_time}_i + \frac{\text{output\_len}_i}{\text{DECODE\_RATE}} .
$$

Request $i$ **interferes** if any *other* request $j$'s prefill window
$[\text{arrival}_j,\ \text{arrival}_j + \text{prefill\_dur}_j]$ overlaps
$i$'s decode window $[\text{ttft\_time}_i,\ \text{decode\_end}_i]$.

- **Colocated**: $\text{TTFT}_i = \text{prefill\_dur}_i$;
  $\text{ITL}_i = \frac{\text{INTERFERENCE\_FACTOR}}{\text{DECODE\_RATE}}$ if
  request $i$ interferes, else $\frac{1}{\text{DECODE\_RATE}}$.
- **Disaggregated**: $\text{TTFT}_i = \text{prefill\_dur}_i +
  \text{TRANSFER\_LATENCY}$; $\text{ITL}_i = \frac{1}{\text{DECODE\_RATE}}$
  always (no interference — separate engines).

Return `(goodput_colocated, goodput_disaggregated)` as plain Python
`int`s — the count of requests meeting both SLOs under each architecture.

## Example

A request with a tight `itl_slo` that happens to overlap another request's
prefill will FAIL under colocated serving (its ITL gets inflated by
`INTERFERENCE_FACTOR`) but PASS under disaggregated serving — while a
different request with a very tight `ttft_slo` might PASS under colocated
(no transfer overhead) but FAIL under disaggregated once
`TRANSFER_LATENCY` pushes its TTFT over the threshold. The two goodput
numbers capture exactly this trade-off.

## What the gate checks

A single gate, **exact_match**, compares your returned `(goodput_colocated,
goodput_disaggregated)` tuple against an independently computed reference
on a fixed, deterministic 30-request trace (`np.random.default_rng(0)`:
Poisson-ish arrivals, varied prompt/output lengths, and a mix of tight and
loose SLOs). Both counts must match exactly.
