# Who's responsible for the 12-second cold request

Production alert: the first request to a local runner after an idle period
sometimes takes 12 seconds, while a normal request takes 400 ms. The runner's
logs only have raw event timestamps (`daemon_start`, `load_end`,
`prefill_start`, ...), and nobody can quickly say how much of those 12
seconds went into bringing up the daemon, how much into loading the weights
into memory, and how much into the actual generation. Because of this, every
incident gets investigated by hand, staring at raw logs, and half the time
the wrong component gets blamed.

We need to turn the raw event log of a single request into a clear
phase-by-phase breakdown.

## What you write

`coldpath/phases.py` — `build_phases(events) -> list[phase]`. `events` is a
list of `{"name", "t"}` (they may appear in the log in any order). There are
exactly four phases: `daemon`, `load`, `prefill`, `decode`. A phase makes it
into the result only if the log has **both** of its boundaries —
`<name>_start` and `<name>_end`; if a boundary is missing (for example the
daemon was already running and wasn't brought up), that phase is simply
absent from the result. A phase is `{"name", "start", "end", "duration_ms"}`.
The list is ordered in the fixed order `daemon, load, prefill, decode`
(only the phases that are present).

`coldpath/attribution.py`:

```python
total_wall_clock(events)      # request_out.t - request_in.t
phase_breakdown(events)       # {name: duration_ms} from build_phases
classify_request(events)      # "cold" | "warm_daemon" | "hot"
unattributed_ms(events)       # total_wall_clock minus the sum of phases
```

`classify_request`: if `daemon` is among the phases, the request is
`"cold"`; if there's no `daemon` but there is `load`, it's `"warm_daemon"`
(the daemon was alive, the model got swapped); if neither is present, it's
`"hot"`.

`coldpath/timeline.py` — `cumulative_ms(events, checkpoints)`. For each
offset in ms from `request_in` in the `checkpoints` list (in increasing
order), return how many milliseconds have been spent so far **inside any of
the named phases**: a phase that started earlier and hasn't ended yet counts
partially (only the elapsed portion). The number must never decrease as the
checkpoint grows.

## How it's graded

The grader computes the reference itself, from the same events, across
several request scenarios — cold start, model swap on a live daemon, hot
request. The third milestone is yours: you write a test, and we swap
`build_phases` for a version that merges `daemon` and `load` into a single
phase (a classic investigation mistake: "brought up the daemon" and "loaded
the weights" look like one indistinguishable blob on the graph). Your test
has to catch that.

```
mlsys project start m-attribute-a-cold-request-s-wall-clock-across-daemon-
mlsys project grade m-attribute-a-cold-request-s-wall-clock-across-daemon- --milestone 1
```
