## Context

Production constrained decoders often use a finite-state machine (FSM) to decide
which tokens may be emitted at each decoding step. A mask is generated from the
current state and applied before sampling the next token.

Let $S$ be the set of FSM states and let $A(s)$ be the set of tokens allowed by
the transitions leaving state $s$. An end-of-sequence token $\mathrm{EOS}$ should
not be accepted just because it is always available. It is valid only when the
current state is accepting.

The allowed token set is therefore

$$
M(s) =
\begin{cases}
A(s) \cup \{\mathrm{EOS}\}, & \text{if } s \text{ is accepting},\\
A(s), & \text{otherwise}.
\end{cases}
$$

A common bug is adding $\mathrm{EOS}$ to every mask. This lets generation stop
before the FSM has reached a valid completed sequence.

## Task

Implement `allowed_tokens(states, transitions, accepting, eos)`:

```python
def allowed_tokens(states, transitions, accepting, eos):
    ...
```

The arguments are:

- `states`: a list of state names in the FSM.
- `transitions`: a dictionary mapping each state name to a dictionary of
  `token -> next_state` transitions.
- `accepting`: a set of accepting state names.
- `eos`: the special EOS token.

Return a dictionary mapping every state name to a `set` of tokens that are valid
in that state. Include EOS only for states contained in `accepting`.

The function should not mutate its inputs.

## Example

```python
states = ["start", "value", "done"]
transitions = {
    "start": {"A": "value"},
    "value": {"B": "done"},
    "done": {}
}
accepting = {"done"}

allowed_tokens(states, transitions, accepting, "<EOS>")

# {
#   "start": {"A"},
#   "value": {"B"},
#   "done": {"<EOS>"}
# }
```

## What the gate checks

The gate builds several FSMs and computes the expected masks with an independent
FSM traversal algorithm. The returned allowed-token sets must exactly match the
oracle output for every state.

The `exact_match` score must be $1.0`. A mask that allows EOS in non-accepting
states fails because it changes which partial strings can terminate.
