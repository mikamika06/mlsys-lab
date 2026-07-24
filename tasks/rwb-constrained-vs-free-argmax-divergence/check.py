import numpy as np


def _step_argmaxes(logits_t, allowed_tokens, vocab_size):
    """Real masked-logits computation, matching how a grammar/FSM-constrained
    decoder (outlines, xgrammar, lm-format-enforcer, ...) actually applies a
    token mask: set disallowed logits to -inf, then argmax. Ties on the free
    argmax and on the masked argmax both resolve to the lowest token id,
    since np.argmax returns the first occurrence."""
    logits_t = np.asarray(logits_t, dtype=np.float64)
    free = int(np.argmax(logits_t))

    mask = np.full(vocab_size, -np.inf, dtype=np.float64)
    allowed_tokens = sorted(set(int(t) for t in allowed_tokens))
    mask[allowed_tokens] = logits_t[allowed_tokens]
    constrained = int(np.argmax(mask))
    return free, constrained


def _oracle(logits, trace, allowed):
    logits = np.asarray(logits, dtype=np.float64)
    vocab_size = logits.shape[1]
    count = 0
    for t, state in enumerate(trace):
        free, constrained = _step_argmaxes(logits[t], allowed[state], vocab_size)
        if free != constrained:
            count += 1
    return count


def _hand_cases():
    cases = []

    # Every step's allowed set is the full vocab -> never diverges.
    rng = np.random.default_rng(11)
    logits = rng.standard_normal((5, 8))
    allowed = {0: list(range(8))}
    cases.append((logits, [0, 0, 0, 0, 0], allowed))

    # Allowed set is a single fixed token each step -> constrained argmax is
    # forced to that token; diverges whenever the free argmax isn't it.
    rng = np.random.default_rng(12)
    logits = rng.standard_normal((6, 10))
    allowed = {0: [3]}
    cases.append((logits, [0] * 6, allowed))

    # Hand-built logits/mask where the top-1 is excluded on some steps and
    # included on others, mixing FSM states.
    logits = np.array(
        [
            [5.0, 1.0, 2.0, 0.0],  # top-1 = 0
            [1.0, 5.0, 2.0, 0.0],  # top-1 = 1
            [1.0, 2.0, 5.0, 0.0],  # top-1 = 2
            [1.0, 2.0, 0.0, 5.0],  # top-1 = 3
        ]
    )
    allowed = {
        0: [0, 1],  # step0 state0: top-1 (0) allowed -> no divergence
        1: [0, 2],  # step1 state1: top-1 (1) excluded -> divergence
        2: [2, 3],  # step2 state1: top-1 (2) allowed -> no divergence
    }
    cases.append((logits, [0, 1, 1, 1], allowed))

    return cases


def _gen_case(rng, T=12, vocab=40, n_states=3):
    logits = rng.standard_normal((T, vocab))
    allowed = {}
    for s in range(n_states):
        k = int(rng.integers(2, vocab // 2))
        allowed[s] = list(rng.choice(vocab, size=k, replace=False))
    trace = [int(rng.integers(0, n_states)) for _ in range(T)]
    return logits, trace, allowed


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = _hand_cases()
    for _ in range(8):
        cases.append(_gen_case(rng))

    exact = 1.0
    for logits, trace, allowed in cases:
        ref = _oracle(logits, trace, allowed)
        try:
            got = sol.constrained_free_argmax_divergence(
                np.array(logits, dtype=np.float64, copy=True),
                list(trace),
                {k: list(v) for k, v in allowed.items()},
            )
        except Exception:
            exact = 0.0
            break
        try:
            got_int = int(got)
        except Exception:
            exact = 0.0
            break
        if got_int != ref:
            exact = 0.0
            break

    return {"exact_match": exact}
