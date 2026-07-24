## Context

Grammar-constrained decoding libraries (`outlines`, `lm-format-enforcer`,
and similar tools used to force an LLM to emit valid JSON, valid code, or a
custom DSL) compile a regular grammar into a **deterministic finite-state
machine (DFA)** and, at generation time, simply walk that machine one
character (or token) at a time to check whether a candidate string is still
on a valid path. Formally, a DFA is a tuple
$(\text{transitions}, q_0, F)$: a transition function
$\delta(q, c) \to q'$, a start state $q_0$, and a set of accepting states
$F$. A string $s = c_1 c_2 \dots c_n$ is **accepted** iff repeatedly
applying $\delta$ from $q_0$ ends in a state in $F$:

$$
q_0 \xrightarrow{c_1} q_1 \xrightarrow{c_2} q_2 \xrightarrow{\cdots} q_n \in F .
$$

If $\delta(q_i, c_{i+1})$ is undefined for some prefix, the machine has no
valid continuation from there — it is **stuck**, and the string is
rejected outright (no need to look at the rest of it).

## Task

Implement `run_fsm`:

```python
def run_fsm(transitions: dict, start_state, accept_states: set, s: str) -> bool:
    ...
```

* `transitions` — `dict` mapping `(state, char) -> next_state`.
* `start_state` — the machine's initial state.
* `accept_states` — a `set` of accepting states.
* `s` — the candidate string to test.

Starting at `start_state`, consume `s` one character at a time, looking up
`transitions[(state, char)]` at each step. If a lookup misses, the machine
is stuck — return `False` immediately. After the whole string has been
consumed (or immediately, for the empty string), return `True` iff the
current state is in `accept_states`.

## Example

```python
# DFA over the empty alphabet {"0", "1"}: state = (value read so far) mod 3.
# Accepts exactly the binary encodings of non-negative multiples of 3.
transitions = {}
for r in range(3):
    transitions[(r, "0")] = (2 * r) % 3
    transitions[(r, "1")] = (2 * r + 1) % 3
start_state, accept_states = 0, {0}

run_fsm(transitions, start_state, accept_states, "0110")   # "0110" = 6  -> True
run_fsm(transitions, start_state, accept_states, "1")      # "1" = 1    -> False
run_fsm(transitions, start_state, accept_states, "")       # "" = 0     -> True
```

## What the gate checks

A single gate, **exact_match**, runs your `run_fsm` over the mod-3-binary
grammar above against a fixed fixture of 49 candidate strings
(`strings.npy` — mostly random-length random bit strings, plus a few
hand-picked edge cases including the empty string) and compares every
accept/reject label against an independently computed reference. All 49
labels must match exactly.
