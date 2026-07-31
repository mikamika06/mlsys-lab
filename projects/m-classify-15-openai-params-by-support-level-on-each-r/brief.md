# Which local runners actually support which parameter

We have several local runners sitting behind one `/v1/chat/completions`
endpoint (each one serves its own pool of models), and the client code is
the same everywhere — plain OpenAI SDK. Complaints come in at random: one
client's `seed` stopped giving reproducibility after we switched them to a
different pool, another client's `tools` are silently never invoked even
though the response comes back `200` and looks fine on the surface. There's
also a separate complaint from ops: the runner's native admin panel shows
queue depth and cache hit rate, but the monitoring that only reads `usage`
from the shim's response never sees those numbers at all — which is exactly
what would let you catch a problem in advance instead of after the fact.

We need to write down, once and honestly, which parameter actually works on
which runner, and which one is just silently swallowed or stripped out by
validation. And separately — which counters each runner tracks natively,
and which of those get lost once the response passes through the
OpenAI-compatible shim.

## What you're writing

`oaicompat/params.py` — `classify_params(runner) -> list[dict]`. `runner`
is `{"name", "supported": set[str], "ignored": set[str], "native_counters": [...]}`.
There's a fixed list of 15 parameters, `PARAMS` (already in the file). For
each parameter in `PARAMS`, in name-sorted order:

- it's in `runner["supported"]` → level `"supported"`;
- otherwise it's in `runner["ignored"]` → level `"ignored"`;
- otherwise → level `"unsupported"` (the runner doesn't know this field at
  all: passing it will either fail validation, or at best get ignored with
  no guarantees whatsoever).

The result is a list of `{"param": ..., "level": ...}`, one entry per each
of the 15 parameters.

`oaicompat/counters.py`:

```python
native_counters(runner)   # every counter the runner tracks natively
shim_counters(runner)     # only the ones that make it into OpenAI's standard usage field
hidden_counters(runner)   # native_counters minus shim_counters
```

The standard OpenAI `usage` schema is exactly `OPENAI_USAGE_FIELDS` (already
in the file): `completion_tokens`, `prompt_tokens`, `total_tokens`.
`shim_counters` is the intersection of the runner's native counters with
this fixed set: the shim never shows the client anything beyond these three
fields, no matter how much of its own telemetry the runner keeps internally.
All three functions return sorted lists with no duplicates.

## How it's graded

The grader computes the reference answer itself, on its own set of runners.
The third milestone is yours: you write a test, and we swap `shim_counters`
for a version that hands the client absolutely everything the runner sees
natively. Your test has to catch that.

```
mlsys project start m-classify-15-openai-params-by-support-level-on-each-r
mlsys project grade m-classify-15-openai-params-by-support-level-on-each-r --milestone 1
```
