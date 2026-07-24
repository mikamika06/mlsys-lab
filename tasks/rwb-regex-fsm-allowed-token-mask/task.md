## Context

Grammar-constrained decoding libraries compile a regular grammar into a
DFA and, at EVERY generation step, need the set of tokens that keep the
output on a path to eventual acceptance — this is what actually masks the
LLM's logits before sampling. It is **not** simply "every token with a
defined transition from here": a transition can lead straight into a
*dead* (trap) state from which no accepting state is reachable at all, and
such a token must be excluded even though the transition technically
exists.

Formally, given a DFA $(\delta, q_0, F)$ (transition function, start
state, accept states), define the set of **alive** states — those that
can still reach some state in $F$ via zero or more further transitions.
$F \subseteq \text{alive}$ trivially (an accept state "reaches" itself).
A state $s$ is alive iff some accepting state is reachable by walking
transition edges *forward* from $s$ — equivalently, iff $s$ is reachable
walking edges *backward* starting from $F$.

From the decoder's current state $q$, a candidate next token $c$ is
**allowed** iff:

$$
\delta(q, c) \text{ is defined} \quad \text{AND} \quad \delta(q, c) \in \text{alive} .
$$

## Task

Implement `allowed_next_tokens`:

```python
def allowed_next_tokens(transitions: dict, current_state, accept_states: set, vocab: list) -> set:
    ...
```

* `transitions` — `dict` mapping `(state, char) -> next_state`: the
  **complete** DFA (you need the whole machine, not just the current
  state's row, to determine which states are alive).
* `current_state` — the state the decoder is currently in.
* `accept_states` — `set` of accepting states.
* `vocab` — list of candidate next tokens (single characters) to test.

Return the `set` of tokens from `vocab` that are allowed per the
definition above: the transition from `current_state` must exist, and
must land on an alive state (one that can still reach some accept state).

## Example

```python
# Grammar: a*b  (zero or more 'a', then exactly one 'b', then nothing else)
transitions = {
    ('q0','a'): 'q0', ('q0','b'): 'q1', ('q0','c'): 'qtrap',
    ('q1','a'): 'qtrap', ('q1','b'): 'qtrap', ('q1','c'): 'qtrap',
    ('qtrap','a'): 'qtrap', ('qtrap','b'): 'qtrap', ('qtrap','c'): 'qtrap',
}
accept_states = {'q1'}
vocab = ['a', 'b', 'c']

allowed_next_tokens(transitions, 'q0', accept_states, vocab)
# -> {'a', 'b'}   ('c' has a transition, straight into the dead trap state -- excluded)

allowed_next_tokens(transitions, 'q1', accept_states, vocab)
# -> set()        (q1 is already accepting, but EVERY further transition
#                   from it leads to the trap state, so nothing keeps a
#                   path to acceptance alive)
```

## What the gate checks

A single gate, **exact_match**, checks your allowed-token set against a
reference (built by an independent breadth-first search over the reversed
transition graph) at EVERY state visited while walking two different
fixed grammars along a fixed decode prefix each: `a*b` along prefix
`"aab"`, and `(ab)*c` along prefix `"abab"` — both grammars include an
explicit trap state reachable by a technically-valid transition, so a
naive "transition exists" implementation (ignoring reachability) will
produce a wrong set at several steps. Every step's returned set must match
the reference exactly.
