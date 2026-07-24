## Context

Disaggregated LLM serving (DistServe, Mooncake, ...) splits prefill and
decode onto **separate worker pools**, because the two phases have very
different compute profiles (prefill is compute-bound over the whole
prompt; decode is one token at a time, memory-bandwidth bound). A
scheduler routing requests into these pools uses **least-loaded**
assignment: send each phase to whichever worker in that pool will become
free soonest.

For $P$ prefill workers, each worker $p$ tracks $a_p$, the time it becomes
available. When request $i$ (arriving at $t_i$, with $\ell_i$ prompt
tokens) needs prefill, it goes to

$$
p^\star = \arg\min_p a_p
$$

(ties broken by the lowest worker index), starts at
$\max(a_{p^\star}, t_i)$, and finishes at

$$
f_i = \max(a_{p^\star}, t_i) + \ell_i \cdot \tau_{\text{prefill}}
$$

after which $a_{p^\star}$ is updated to $f_i$. Decode can't start until
prefill hands off the KV cache, so request $i$'s decode phase becomes
biddable at $f_i$: it's routed to whichever of the $D$ decode workers has
the smallest availability $b_d$, by the exact same rule, using $f_i$ (not
$t_i$) as its earliest possible start and $g_i \cdot \tau_{\text{decode}}$
($g_i$ = tokens to generate) as its duration.

## Task

Implement `route_prefill_decode(arrivals, prompt_lens, gen_lens, n_prefill_workers, n_decode_workers, t_prefill_per_token, t_decode_per_token)`:

```python
def route_prefill_decode(arrivals, prompt_lens, gen_lens, n_prefill_workers, n_decode_workers, t_prefill_per_token, t_decode_per_token):
    ...
```

Requests are handled **one at a time, in the given order** (this order is
the arrival order — `arrivals` is already sorted). For each request,
route its prefill phase and then its decode phase exactly as described
above, updating each pool's worker-availability state as you go.

Return `(prefill_assignments, decode_assignments)`:

- `prefill_assignments[p]`: list of request indices routed to prefill
  worker `p`, in the order they were assigned.
- `decode_assignments[d]`: list of request indices routed to decode
  worker `d`, in the order they were assigned.

## Example

```python
route_prefill_decode(
    arrivals=[0.0, 0.0, 0.0, 0.0],
    prompt_lens=[100, 100, 100, 100],
    gen_lens=[10, 10, 10, 10],
    n_prefill_workers=2, n_decode_workers=2,
    t_prefill_per_token=0.01, t_decode_per_token=0.02,
)
# requests 0,1,2,3 all arrive at t=0 with identical prompt/gen lengths.
# Prefill: worker 0 takes req 0 (finish 1.0), worker 1 takes req 1
#   (finish 1.0) -- both tied at 0.0 before assignment, lowest index
#   wins first. Then both are back at 1.0, so req 2 -> worker 0
#   (finish 2.0), req 3 -> worker 1 (finish 2.0).
# -> prefill_assignments = [[0, 2], [1, 3]]
# Decode assignment follows the same least-loaded logic using each
# request's own prefill-finish time as its decode arrival.
```

## What the gate checks

The gate runs several hand-built scenarios (identical simultaneous
arrivals splitting evenly across workers, a single slow prefill blocking
one worker while others race ahead with small requests, more pools than
requests, and an empty trace) plus randomly generated traces with varying
pool sizes, arrival times, prompt/gen lengths, and per-token service
times from a seeded generator.

For every case the reference runs the exact greedy simulation described
above and returns the two assignment-list structures. Your output is
compared to it with exact equality — every worker's list, in order, on
every case. A solution that decides the decode-phase routing using each
request's original *arrival* time instead of its actual *prefill-finish*
time will match whenever prefill is instantaneous or workers are never
staggered, but disagree with the oracle as soon as prefill queueing
delays make requests finish prefill in a different order than they
arrived.
