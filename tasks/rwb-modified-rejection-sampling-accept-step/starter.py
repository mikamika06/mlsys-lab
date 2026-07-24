import numpy as np


def modified_rejection_sample(p, q, draft_token_ids, u_stream):
    """
    p, q: (T, V) target and draft distributions, each row summing to 1.
    draft_token_ids: (T,) int array, drafted token id per position.
    u_stream: 1-D float array of pre-drawn uniforms in [0,1), consumed
        sequentially from the front with a single shared pointer: one
        draw for the accept check at every position, plus one more only
        when that position is rejected.

    For each position t in order: accept draft_token_ids[t] iff the next
    draw <= min(1, p[t, tok]/q[t, tok]); otherwise pop one more draw and
    emit a residual sample from normalize(max(p[t]-q[t], 0)) via inverse
    CDF (np.searchsorted(cumsum(r), u, side="left"), clipped to V - 1).

    Returns an int64 NumPy array of shape (T,): the emitted token id at
    each position.
    """
    raise NotImplementedError('your code here')
