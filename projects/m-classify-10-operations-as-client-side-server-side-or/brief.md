# Client, server, or runner: where does each operation belong

Our local grading runner right now is one process doing everything: it accepts
the CLI request, runs the learner's solution immediately, and writes the result
to disk itself. Last week someone's infinite loop in a solution hung this
process for 40 seconds — and every other submission in the queue sat waiting
for it to come back, even though they had nothing to do with that code. Then
after the daemon restarted, the whole queue of ungraded jobs vanished: state
lived nowhere but the process's memory, which comes up fresh every time.

The symptom is one thing: everything is mixed together in one place. We need
to split the system's operations across three sides — client (the user's
CLI), server (the long-lived daemon with the queue), and runner (the isolated
process that executes someone else's code) — by an explicit rule that can be
checked automatically, not just kept in someone's head.

Here are the 10 operations that currently run wherever they happen to land:

1. `parse_cli_arguments` — parse the command-line arguments.
2. `render_progress_output` — draw progress for the user in the terminal.
3. `read_local_skeleton_file` — read the learner's file from local disk.
4. `enqueue_grading_job` — put a job on the grading queue.
5. `persist_job_status` — write job status so it survives a restart.
6. `route_logs_to_waiting_client` — deliver a log line to the client waiting for it.
7. `execute_learner_solution` — run the learner's solution.
8. `import_and_call_learner_module` — import the learner's module and call its function (the checker does this itself).
9. `enforce_process_timeout` — kill the child process if it exceeds the time limit.
10. `measure_peak_child_memory` — measure that same child process's peak memory usage.

## What you write

`opside/classify.py` — `classify_operation(op) -> str` and `classify_all(config) -> list[dict]`.

An operation is `{"name", "runs_untrusted_code", "needs_sandbox_proximity", "needs_durable_state"}`,
the last three fields boolean. `needs_sandbox_proximity` means: the operation
itself doesn't run untrusted code, but it has to live right next to the
isolated process (kill it on timeout, read the child's metrics) — anywhere
else it's either pointless or unsafe.

Classification rule, in priority order:

1. If `runs_untrusted_code` or `needs_sandbox_proximity` — `"runner"`.
2. Else if `needs_durable_state` — `"server"`.
3. Otherwise — `"client"`.

A config is `{"operations": [op, ...]}`. `classify_all(config)` returns a list
of `{"name", "side"}` in the same order as the input operations.

`opside/pipeline.py`:

```python
side_sequence(config, order) -> list[str]
hop_count(config, order) -> int
is_legal_pipeline(config, order) -> bool
```

`order` is the list of operation names in the order they run for a single
request (a name can repeat several times). `side_sequence` returns the side
of each step in turn. `hop_count` counts how many times the side changed from
one step to the next. `is_legal_pipeline` is `False` if `"client"` and
`"runner"` ever sit directly next to each other in the sequence with no
server between them; otherwise `True`.

## How it's graded

The grader computes the reference itself — across several operation configs
(including the full enumeration of all eight combinations of the three
flags) and several request execution sequences. The third milestone is
yours: you write a test, and we swap in a classification that only looks at
`needs_durable_state` and ignores the need for isolation — i.e. one that lets
untrusted code execution land straight on the client or the server. Your
test needs to catch that.

```
mlsys project start m-classify-10-operations-as-client-side-server-side-or
mlsys project grade m-classify-10-operations-as-client-side-server-side-or --milestone 1
```
