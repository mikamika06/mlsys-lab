## Context

In disaggregated prefill/decode serving (Mooncake, DistServe, ...), a
request's KV cache is computed on a prefill worker and then shipped over
the network to a decode worker. Request $i$'s transfer moves
$B_i = \text{num\_tokens}_i \cdot \text{kv\_bytes\_per\_token}$ bytes at a
constant rate over its transfer window
$[s_i,\, s_i + d_i)$ (start = the moment its prefill finishes, duration
$d_i$):

$$
\text{rate}_i = \frac{B_i}{d_i}
$$

Many requests finish prefill around the same time and transfer
concurrently, so the network link has to sustain the **sum** of every
currently-active transfer's rate. The number that determines whether a
link is provisioned correctly isn't the average bandwidth — it's the
**peak** instantaneous demand:

$$
\text{peak} = \max_{t} \sum_{i \,:\, s_i \le t < s_i + d_i} \text{rate}_i
$$

## Task

Implement `peak_kv_transfer_bandwidth(prefill_complete_times, num_tokens, transfer_durations, kv_bytes_per_token)`:

```python
def peak_kv_transfer_bandwidth(prefill_complete_times, num_tokens, transfer_durations, kv_bytes_per_token) -> float:
    ...
```

- `prefill_complete_times[i]`: the time request $i$'s KV transfer begins.
- `num_tokens[i]`: number of tokens in request $i$'s KV cache.
- `transfer_durations[i]`: how long request $i$'s transfer takes.
- `kv_bytes_per_token`: bytes of KV cache per token (same for every
  request).

Return the peak aggregate bandwidth demanded at any single instant, as
defined above. Each transfer's window is **half-open**: a transfer ending
at exactly $t$ does not overlap one starting at exactly $t$.

## Example

```python
peak_kv_transfer_bandwidth(
    prefill_complete_times=[0.0, 0.0, 0.0],
    num_tokens=[100, 200, 50],
    transfer_durations=[2.0, 2.0, 2.0],
    kv_bytes_per_token=4.0,
)
# all three transfers fully overlap for their entire 2.0s window ->
# peak = sum of the three rates = (400 + 800 + 200) / 2.0 = 700.0
```

## What the gate checks

The gate runs several hand-built traces (disjoint transfers where the
peak is just the largest single rate, fully overlapping transfers where
the peak is the sum of all rates, staggered partial overlaps, an "exact
touch" case where one transfer ends the instant another starts, and a
single request) plus randomly generated traces with varying start times,
durations, token counts, and `kv_bytes_per_token` from a seeded generator.

For every trace the reference sweeps the events in time order — with
"end" events applied before "start" events whenever two coincide exactly,
matching the half-open window semantics — tracking the running sum of
active rates and its maximum. Your return value is compared to that peak
with relative error, requiring `rel_err < 1e-9`. A solution that instead
just sums *all* rates whose window contains the maximum of the start
times (a single fixed instant) rather than genuinely sweeping every event
time will miss cases where the true peak occurs at a different instant,
or where a transfer overlapping every start time isn't actually the
busiest point once shorter transfers are considered.
